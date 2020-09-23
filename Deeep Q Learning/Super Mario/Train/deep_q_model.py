import time
import gym
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from priority_replay import Memory
import atari_wrapper
from annealing_variable import AnnealingVariable
from stats import Stats
import atari_wrapper
import pandas as pd
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# the hubber loss is used instead of mse
def huber_loss(y_true, y_pred, delta=2.0):
    err = y_true - y_pred

    cond = K.abs(err) < delta

    L2 = 0.5 * K.square(err)
    L1 = delta * (K.abs(err) - 0.5 * delta)

    # if error < delta perform mse error else perform mae
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


class DQNAgent:
    def __init__(self):
        self.action_space = env.action_space.n # number of actions
        self.discount_rate = 0.99
        self.stack_frames = 4  # number of frames stacked
        self.lr = 2.5e-5
        self.model = self.build_deep_q_model()  # build the main network
        self.target_model = self.build_deep_q_model()   # build the target network
        mask_shape = (1, env.action_space.n)
        self.batch_size = 32  # number of batches to learn from
        self.priority_replay_size = 1000000   # max replay memory size

        self.priority_replay = Memory(self.priority_replay_size)   # create the replay memory
        self.exploration_rate = AnnealingVariable(1., .01, 400000)  # exploration rate (start_value, final_value, n_steps)
        self.mask = np.ones(mask_shape, dtype=np.float32) # mask used in the training to train only the q values for which the agent performed and action
        self.one_hot = np.eye(env.action_space.n, dtype=np.float32)

        self.update_target_frequency = 10000  # update target every
        self.replay_frequency = 4  # learn from memories frequency

    # creates the network and returns it
    def build_deep_q_model(self,
        image_size: tuple = (84, 84)
    ) -> keras.Model:
        # weights initializer - he init
        initializer = keras.initializers.VarianceScaling(scale=2.0)
        # input layer - takes (84,84,84,4) images
        cnn_input = keras.layers.Input((*image_size, self.stack_frames), name="cnninpt")
        # mask input - one hot when want only q_value for particular action
        mask_input = keras.layers.Input((self.action_space,), name="mask")

        # first Conv2D layer
        cnn = keras.layers.Conv2D(32, 8, strides=4, activation="relu", padding='valid', kernel_initializer=initializer)(cnn_input)
        # second Conv2D layer
        cnn = keras.layers.Conv2D(64, 4, strides=2, activation="relu", padding='valid', kernel_initializer=initializer)(cnn)
        # third Conv2D layer
        cnn = keras.layers.Conv2D(64, 3, strides=1, activation="relu", padding='valid', kernel_initializer=initializer)(cnn)

        # flatten the kernels from previous layers
        cnn = keras.layers.Flatten()(cnn)

        # fully connected layer
        cnn = keras.layers.Dense(512, activation="relu", kernel_initializer=initializer)(cnn)
        # output layer, q_values for every action in enviroment
        cnn = keras.layers.Dense(self.action_space, name="output")(cnn)

        # multiply output by mask to give true output
        output = keras.layers.Multiply()([cnn, mask_input])

        # create the model
        model = keras.Model(inputs=[cnn_input, mask_input], outputs=output)

        # add loss function and method of optimization
        model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(lr=self.lr))
        print(model.summary())
        return model

    # samples a batch of #batch_size from the priority replay
    def sample(self):
        batch = self.priority_replay.sample(self.batch_size) # sample batch according to priorities
        X_state = [None] * len(batch)   # create empty list #batch_size
        X_action = [None] * len(batch)
        X_reward = [None] * len(batch)
        X_done = [None] * len(batch)
        X_next_state = [None] * len(batch)
        # for each batch - retrieve the particualar entries
        for i in range(len(batch)):
            o = batch[i][1]
            X_state[i] = np.array(o[0], copy=False)
            X_action[i] = o[1]
            X_reward[i] = o[2]
            X_done[i] = o[3]
            X_next_state[i] = np.array(o[4], copy=False)

        return np.array(X_state), np.array(X_action), np.array(X_reward), np.array(X_done), np.array(X_next_state), batch

    # train on the samples in memory
    def learn_from_memories(self):
        X_state, X_action, X_reward, X_done, X_next_state, batch = self.sample()

        # repeat the base mask (all actions) for all the samples in the batch
        mask = np.repeat(self.mask, len(X_state), axis=0)

        # q for the next_state is 0 if episode has ended or else the max of q values according to the target network
        q_next = np.max(self.target_model.predict_on_batch([X_next_state, mask]), axis=1)
        q_next[X_done] = 0.0

        # the q predictions on batch
        pred = self.model.predict_on_batch([X_state, self.mask])

        # the q predictions for each action taken only
        pred_action = pred[range(pred.shape[0]), X_action]

        # calculate the target / true values
        target_q = X_reward + self.discount_rate * q_next

        # get the error - priorirty
        error = abs(target_q - pred_action)

        # update errors - priorities
        for i in range(len(batch)):
            idx = batch[i][0]
            self.priority_replay.update(idx, error[i])

        # assign target_q to the appropriate true_q columns for which the agent performed an action
        # all other true q values are 0 and the agent is not trained based on their value
        true_q = np.zeros((len(X_state), env.action_space.n), dtype=np.float32)
        # for every row, for the columns the agent picked that action assign target_q
        true_q[range(true_q.shape[0]), X_action] = target_q
        # train only on actions the agent chose - One hot mask from action batch
        return self.model.fit([X_state, self.one_hot[X_action]], true_q, verbose=0, epochs=1)

    # helper function to initialize the priority replay with init_size samples
    def init_priority_replay(self, init_size=50000):
        while init_size > 0:
            done = False
            state = env.reset() # reset the env
            while not done:
                # behave randomly
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                # save batch of experience to memory
                self.save_to_memory(state, action, reward, done, next_state)
                state = next_state
                init_size -= 1

    # takes the experience and stores it in replay memory
    def save_to_memory(self, state, action, reward, done, next_state):
        error = self.get_error(state, action, reward, done, next_state) # find error - priority
        self.priority_replay.add(error, (state, action, reward, done, next_state)) # save to memory according to priority

    # behaves epsilon greedily
    def epsilon_greedy_policy(self, state, epsilon=0.01):
        if np.random.random() < epsilon:
            return env.action_space.sample()  # pick randomly

        q_values = self.predict_qvalues(state)
        return np.argmax(q_values)  # pick optimal action - max q(s,a)

    # calculate the error of prediction
    def get_error(self, state, action, reward, done, next_state):
        # q next is 0 for terminal states else predict it from target network
        if done:
            q_next = 0.0
        else:
            next_state = np.expand_dims(next_state, axis=0)
            q_next = self.target_model.predict([next_state, self.mask])

        predicted_q = self.predict_qvalues(state)
        # error = true - predicted for action taken
        error = abs(reward + self.discount_rate*np.max(q_next) - predicted_q[0, action])
        return error

    # predicts and returns q values from state
    def predict_qvalues(self, state: np.ndarray):
        # keras needs first dimension to be batch size even if only one sample
        state = np.expand_dims(state, axis=0)
        return self.model.predict([state, self.mask])

    # trains the agent for max_steps
    def train_model(self, max_steps, stats):
        print("Start Training ")
        while max_steps > 0:
            r_per_episode = 0
            error = 0
            done = False

            state = env.reset()

            while not done:  # the episode has not ended
                # find action according to epsilon greedy behaviour with epsilon = exploration rate
                action = self.epsilon_greedy_policy(state, self.exploration_rate.value)
                # decrease the exploration rate
                self.exploration_rate.step()

                max_steps -= 1
                # take the action and observe next observation, reward, done
                next_state, reward, done, _ = env.step(action)

                r_per_episode += reward

                # save this experience to memory
                self.save_to_memory(state, action, reward, done, next_state)

                state = next_state

                # time to train from priority replay
                if max_steps % self.replay_frequency == 0:
                    hist = self.learn_from_memories()
                    error += hist.history['loss'][0]

                # time to update the target network
                if max_steps % self.update_target_frequency == 0:
                    self.target_model.set_weights(self.model.get_weights())
            #  update stats
            stats(self, r_per_episode)
        # save stats at the end of training
        stats.save_stats()

    # test the agent for nepisodes
    def play(self, nepisodes, exploration_rate, stats):
        rewards_arr = np.zeros(nepisodes)
        for episode in range(nepisodes):
            episode_reward = 0
            done = False
            state = env.reset()
            while not done:
                env.render()
                # time.sleep(0.05)
                action = self.epsilon_greedy_policy(state, exploration_rate)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            rewards_arr[episode] = episode_reward
            stats(self, episode_reward)
            print(episode_reward)
        stats.save_stats()
        return rewards_arr

    def save_weights(self):
        print("PRINT")
        self.model.save_weights("MarioDQNWeights.h5")

    def save_model(self):
        self.model.save("DqnMarioModel.h5")

    def restore_weights(self):
        print("Restoring model weights Mario")
        self.model.load_weights("MarioDQNWeights.h5")


stats = Stats()
_env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
_env = JoypadSpace(_env, SIMPLE_MOVEMENT)
env = atari_wrapper.wrap_dqn(_env)
agent = DQNAgent()
agent.init_priority_replay(50000)
agent.train_model(max_steps=50e6, stats=stats)
env.close()
