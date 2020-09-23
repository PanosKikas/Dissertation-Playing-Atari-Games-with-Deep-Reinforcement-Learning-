import tensorflow as tf
from tensorflow.python.keras import initializers, callbacks
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Multiply, MaxPool2D
import numpy as np

# initializes the network with normalized values
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class Actor_Critic(tf.keras.Model):

    # network for cartpole
    def __init__(self, num_actions):
        super(Actor_Critic, self).__init__()
        # initializer used for the weights - he init
        self.initializer = tf.initializers.variance_scaling()
        # 1 hidden layer for carpole env, 24 neurons
        self.dense = Dense(24, activation="relu", kernel_initializer=self.initializer)
        self.num_actions = num_actions
        # logits for each probability. If log is applied to the logits it gives the probabilities according to softmax activation
        self.logits = Dense(self.num_actions, kernel_initializer=normalized_columns_initializer(0.01))
        # value function approximator
        self.value = Dense(1, activation='linear', kernel_initializer=normalized_columns_initializer(1.0))

    # override call method of model keras class
    def call(self, inputs):
        # pass the input through the first dense layer
        x = self.dense(inputs)
        # take the logits from output of dense
        logits = self.logits(x)
        # take value from output of dense
        value = self.value(x)
        return logits, value
