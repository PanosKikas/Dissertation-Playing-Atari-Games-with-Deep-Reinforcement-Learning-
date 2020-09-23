import threading
import time
import numpy as np
import gym
from annealing_variable import AnnealingVariable
from stats import Stats
import tensorflow as tf
from actor_critic import Actor_Critic
from keras.models import load_model
import atari_wrapper

tf.enable_eager_execution()
env_name = 'Pong-v0'


class A3CAgent(object):
    def __init__(self):
        self.num_actions = NUM_ACTIONS  # number of actions
        self.starter_lr = 1e-4  # start value of learning rate

        # optimizer that trains the global network with the gradients of the locals
        # use locking because multiple threads
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.starter_lr, use_locking=True)
        # the global Actor-Critic network
        self.global_network = Actor_Critic(self.num_actions)
        # prepare the global network - used to construct the network on eager execution
        self.global_network(tf.convert_to_tensor(np.random.random((1, 84, 84, 4)), dtype=tf.float32))
        self.restore_weights()

    def pick_action(self, state, exploration_rate = 0.0):
        if np.random.random() < exploration_rate:
            return test_env.action_space.sample()  # pick randomly

        state = np.expand_dims(state, axis=0)
        logits, _ = self.global_network(state)
        probs = tf.nn.softmax(logits)
        action = np.random.choice(self.num_actions, 1, p=probs.numpy()[0])
        return action[0]

    def play(self, env, stats, episodes: int = 100, exploration_rate=0.0):
        rewards_arr = np.zeros(episodes)
        for episode in range(episodes):
            episode_reward = 0
            done = False
            state = env.reset()
            while not done:
                env.render()
                # time.sleep(0.05)
                action = self.pick_action(state, exploration_rate)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            if callable(stats):
                stats(self, episode_reward)
            rewards_arr[episode] = episode_reward
            print(episode_reward)
        stats.save_stats()
        return rewards_arr

    def restore_weights(self):
        self.global_network.load_weights('A3CPong.h5')


test_env = gym.make(env_name)
test_env = atari_wrapper.wrap_dqn(test_env)

NUM_ACTIONS = test_env.action_space.n
OBS_SPACE = test_env.observation_space.shape[0]

state = test_env.reset()
state = np.expand_dims(state, axis=0)

stats = Stats()
agent = A3CAgent()
agent.play(test_env, stats)
test_env.close()
