# Dissertation: Playing Atari Games with Deep Reinforcement Learning

With this thesis I try to get an in-depth view of Reinforcement Learning in the field of Machine Learning. The goal is to train an agent on a series of tasks each needing composite goals. The agent was trained on four different Atari environments with the purpose to achieve the optimal return. The return depends on the structure of each environment and the agent should strive to discover the optimal actions that will enable it to achieve the optimal reward.
In this dissertation I present two different Reinforcement Learning algorithms and compare the results that they achieved on four different environments.

The first algorithm is called Deep Q Learning and It strives to approximate the optimal value function and then behave optimally according to it.

The second one is called Actor-Critic and it parametrizes the policy as a function and continuously tries to optimize it. This method envelopes elements from the previous method and so It will be presented in the second part of this thesis.

For each method I describe how they operate from the inside, their advantages and disadvantages and their efficiency on the various tasks.
Last but not least, both algorithms remain unchanged for all the environments, meaning that they are not dependant on the structure of the environment and can be used to achieve any task.

--Requires Tensorflow 1.14 or later
