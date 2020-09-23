# # from deep_q_model import DQNAgent
# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class Stats(object):
    def __init__(self):
        self.episodes = 0
        self.rewards = []
        self.save_freq = 100

    def __call__(self, agent, reward: float):
        self.episodes += 1
        self.rewards.append(reward)
        # self.errors.append(loss)

        if self.episodes % 100 == 0:
            print("EPISODE: ", self.episodes, " Rewards: ", np.mean(self.rewards[-50:]))

        if self.episodes % self.save_freq == 0:
            print("Saving model")
            agent.save_weights()
            self.save_stats()

    def save_stats(self):
        rewards = pd.DataFrame(self.rewards)
        # error = pd.Series(self.errors)
        rewards = pd.concat([rewards], axis=1)
        rewards.columns = ['Reward']
        rewards.index.name = 'Episode'
        rewards.to_csv('rewards_DQNMarioTrain.csv')
        # rewards_error.plot(figsize=(12, 5), subplots=True)
        # plt.savefig('./rewards_error_PacmanDQNPlay.pdf')
        # plt.close()
