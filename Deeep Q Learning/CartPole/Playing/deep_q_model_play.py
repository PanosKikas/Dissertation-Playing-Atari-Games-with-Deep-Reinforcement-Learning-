import time
import gym
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from keras.models import load_model
from stats import Stats

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
        self.model = self.restore_model()
        mask_shape = (1, env.action_space.n)
        self.mask = np.ones(mask_shape, dtype=np.float32)

    # behaves epsilon greedily
    def epsilon_greedy_policy(self, state, epsilon=0.01):
        if np.random.random() < epsilon:
            return env.action_space.sample()  # pick randomly

        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict([state,self.mask])
        return np.argmax(q_values)  # pick optimal action - max q(s,a)

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

    def restore_model(self):
        print("Restoring model CartPole")
        return load_model("DQNCartpoleModel.h5", custom_objects={'huber_loss':huber_loss})


stats = Stats()
env = gym.make('CartPole-v0')
agent = DQNAgent()
agent.play(100, 0.0, stats)
env.close()
