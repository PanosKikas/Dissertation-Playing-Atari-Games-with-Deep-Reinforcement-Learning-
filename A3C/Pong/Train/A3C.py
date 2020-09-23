import threading
import time
import numpy as np
import gym
from annealing_variable import AnnealingVariable
from stats import Stats
import tensorflow as tf
from thread_worker import train_thread
from actor_critic import Actor_Critic
import atari_wrapper

NUM_THREADS = 8  # number of parallel actor-critic threads

tf.enable_eager_execution()
env_name = 'Pong-v0'


class A3CAgent(object):
    def __init__(self):
        self.num_state = OBS_SPACE # observation size
        self.num_actions = NUM_ACTIONS # number of actions
        self.lr = tf.Variable(3e-4) # variable used for decaying learning rate
        self.starter_lr = 1e-4 # start value of learning rate

        # optimizer that trains the global network with the gradients of the locals
        # use locking because multiple threads
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.starter_lr, use_locking=True)

        # the global Actor-Critic network
        self.global_network = Actor_Critic(self.num_actions)
        # prepare the global network - used to construct the network on eager execution
        self.global_network(tf.convert_to_tensor(np.random.random((1, 84, 84, 4)), dtype=tf.float32))

        self.discount_rate = 0.99

    def start_threads(self):
        # max number of episodes
        max_eps = 1e6
        envs = []
        # create 1 local enviroment for each thread
        for _ in range(NUM_THREADS):
            _env = gym.make(env_name)
            env = atari_wrapper.wrap_dqn(_env)
            envs.append(env)
        # create the threads and assign them their enviroment and exploration rate
        threads = []
        for i in range(NUM_THREADS):
            thread = threading.Thread(target=train_thread, daemon=True,
                                      args=(
                                          self, max_eps, envs[i], agent.discount_rate, self.optimizer, stats,
                                          AnnealingVariable(.7, 1e-20, 1000), i))
            threads.append(thread)

        # starts the threads
        for t in threads:
            print("STARTING")
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads] # wait for threads to finish
        except KeyboardInterrupt:
            print("Exiting threads!")

    def save_weights(self):
        print("Saving Weights")
        self.global_network.save_weights("A3CPong.h5")

    def restore_weights(self):
        print("Restoring Weights!")
        self.global_network.load_weights("A3CPong.h5")


_test_env = gym.make(env_name)
test_env = atari_wrapper.wrap_dqn(_test_env)

NUM_ACTIONS = test_env.action_space.n
OBS_SPACE = test_env.observation_space.shape[0]

state = test_env.reset()
state = np.expand_dims(state, axis=0)

stats = Stats()
agent = A3CAgent()
agent.start_threads()
test_env.close()
