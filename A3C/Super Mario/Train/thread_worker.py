import numpy as np
from threading import Lock
from keras.utils import to_categorical
from stats import Stats
from annealing_variable import AnnealingVariable
from keras import Model
from keras import backend as K
from keras import losses
import tensorflow as tf
from actor_critic import Actor_Critic


episodes = 0
lock = Lock()
train_frequency = 20

# the local actor-critic worker
def train_thread(agent, max_eps, env, discount_rate, optimizer,
                 statistics: Stats, exploration_rate: AnnealingVariable, number):

    # create local network and init its weights equal to the global
    local_network = Actor_Critic(env.action_space.n)
    # prepare it (must do this when eager execution is enabled)
    local_network(tf.convert_to_tensor(np.random.random((1, 84, 84, 4)), dtype=tf.float32))
    local_network.set_weights(agent.global_network.get_weights())
    lr_decay_anneal = AnnealingVariable(3e-4, 1e-20, 15e6)
    global episodes # number of total episodes done for all threads

    # local lists for states, rewards and actions
    states, rewards, actions = [], [], []
    while episodes < max_eps:
        r_per_episode = 0.0
        done = False
        step = 0
        state = env.reset()
        # still training
        while not done and episodes < max_eps:
            exploration_rate.step() # decay the exploration rate

            states.append(state) # add the observation/state into the state list
            # find acction to pick according to the probs network and the exploration rate
            action = pick_action(env, local_network, state, exploration_rate.value)
            # do the action and observe the next state, reward and if the episode is over
            next_state, reward, done, _ = env.step(action)
            lr_decay_anneal.step() # decrease the learning rate

            # append the reward experienced in the reward list
            rewards.append(reward)
            # append action taken
            actions.append(action)
            r_per_episode += reward

            step += 1

            # if gathered enough experience or the episode is over -> train on experience gathered
            if step%train_frequency==0 or done:
                # Gradient tape records the gradient during the evaluation of the loss function
                # -> eager execution MUST be enabled to work
                with tf.GradientTape() as tape:
                    # compute loss for each batch of experience
                    loss = compute_loss_from_batch(local_network, states, rewards, actions, done, next_state, discount_rate)
                # rewind the tape and get the gradients of the loss
                # for the weights of the local network (Actor-Critic)
                gradients = tape.gradient(loss, local_network.trainable_weights)
                # used because multiple threads
                lock.acquire()
                agent.lr.assign(lr_decay_anneal.value) # set the lr of the global network equal to the local decayed

                # apply the gradients found from the local network into the global network for the global weights
                optimizer.apply_gradients(zip(gradients, agent.global_network.trainable_weights))
                # update local network with weights of global
                local_network.set_weights(agent.global_network.get_weights())
                lock.release()
                # empty state, reward, action list
                states, rewards, actions = [], [], []

            state = next_state
        with lock:
            # save stats
            if episodes < max_eps:
                episodes += 1
                statistics(agent, r_per_episode)



def compute_loss_from_batch(local_network, state_batch, reward_batch, action_batch, done, next_state, discount_rate):

    # if the last state visited with the action taken led to a terminal state
    # the value of the next state is 0, else the value of the next state is retrieved
    # from the local network's output value
    if done:
        DR = 0.0
    else:
        next_state = tf.expand_dims(next_state, axis=0)
        # tf.cast(next_state, dtype=tf.float32)
        DR = local_network(next_state)[-1].numpy()[0] # predict from local network value of next_state

    # list that will gather the discounted rewards
    discounted_rewards = []
    # traverse list in reverse order. Each Discounted reward is computed as the initial reward + discount rate * Discounted rewards so far
    # in the start the Discounted rewards so far is 0 or value of next state (depends on done) and adds each discounted reward to it
    for reward in reward_batch[::-1]:
        DR = reward + discount_rate * DR # current discounted reward
        discounted_rewards.append(DR) # append current dr to list

    # reverse the order to match the order of the batch
    discounted_rewards.reverse()
    discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards), dtype=tf.float32)

    # predict logits and values from local network and state batch
    logits, values = local_network(tf.convert_to_tensor(state_batch, dtype=tf.float32))

    # Evaluate advantage from discounted rewards and values
    advantage = discounted_rewards - values

    # Calculate probs using softmax on logits
    probs = tf.nn.softmax(logits)

    # compute entropy = softmax_crossentropy given logits and probabilities
    # computed : Entropy = -log(softmax(logits)) * probs
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs, logits=logits)
    entropy = tf.expand_dims(entropy, axis=1)
    # mean entropy loss
    entropy = tf.reduce_mean(entropy)

    # compute policy loss for actions taken and logits from output
    # computed: ploss = -log(softmax(logits))*action_batch -> will 0 all the loss for any other action than those taken
    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_batch, logits=logits)
    policy_loss = tf.expand_dims(policy_loss, axis=1)

    # ploss = -log(softmax(logits))* advantage -> advantage treated as constant
    policy_loss *= tf.stop_gradient(advantage)

    policy_loss = tf.reduce_mean(policy_loss)

    # total policy loss = policy_loss - 0.01 * entropy_loss
    policy_loss = policy_loss - 0.01 * entropy

    # Value loss = advantage^2
    v_loss = advantage ** 2
    v_loss = tf.reduce_mean(v_loss)

    # total loss = ploss + 0.5 * vloss
    total_loss = 0.5 * v_loss + policy_loss
    return total_loss


# return an action according to the logits of the local network and exploration rate
def pick_action(env, local_network, state, exploration_rate):
    if np.random.random() < exploration_rate:
        return env.action_space.sample()  # pick randomly

    state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    logits,_ = local_network(state) # predict logits for current state

    probs = tf.nn.softmax(logits) # apply softmax to the logits
    action = np.random.choice(env.action_space.n, p=probs.numpy()[0]) # take a random action acording to the probs found
    return action
