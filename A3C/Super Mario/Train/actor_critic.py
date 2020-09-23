import tensorflow as tf
from tensorflow.python.keras import initializers, callbacks
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Multiply, MaxPool2D
import numpy as np

# initializer tha gives normalized weights with standard deviation given
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

# the Actor-Critic model - overriders keras models
class Actor_Critic(tf.keras.Model):
    def __init__(self, num_actions):
        super(Actor_Critic, self).__init__()
        self.initializer = tf.initializers.variance_scaling()
        # 1st convolution layer
        self.cnn1 = Conv2D(32, 5, activation="relu", padding='same', kernel_initializer=self.initializer)
        # Apply max pool
        self.mp1 = MaxPool2D(2)

        # 2nd convolution layer
        self.cnn2 = Conv2D(32, 5, activation="relu", padding='same')
        # apply max pool
        self.mp2 = MaxPool2D(2)

        # 3rd convolution layer
        self.cnn3 = Conv2D(64, 4, activation="relu", padding='same')
        # apply max pool
        self.mp3 = MaxPool2D(2)

        # 4th convolution layer
        self.cnn4 = Conv2D(64, 3, activation='relu', padding='same')

        # flatten kernel output
        self.flatten = Flatten()
        # pass it to fully connected
        self.dense = Dense(512, activation="relu", kernel_initializer=self.initializer)
        self.num_actions = num_actions

        # lolgits - used for policy estimation
        self.logits = Dense(self.num_actions, kernel_initializer=normalized_columns_initializer(0.01))
        # value function approximation
        self.value = Dense(1, activation='linear', kernel_initializer=normalized_columns_initializer(1.0))

    # override to call function
    def call(self, inputs):
        # x = apply 1st convolution to input then maxpool
        x = self.mp1(self.cnn1(inputs))
        # x = apply 2n convolution then maxpool
        x = self.mp2(self.cnn2(x))
        # x= apply 3rd convolution then maxpool
        x = self.mp3(self.cnn3(x))
        # apply 4th convolution
        x = self.cnn4(x)
        # flatten output
        x = self.flatten(x)

        # pass to fully connected
        x = self.dense(x)
        # logit output
        logits = self.logits(x)
        # value output
        value = self.value(x)
        return logits, value

