# lenet.py

import tensorflow as tf
import tensorflow.contrib.eager as tfe

layers = tf.keras.layers

class Net(tf.keras.Model):
    """docstring for Net"""
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.nc = args.nchannels
        self.noutputs = args.noutputs
        # self.input_shape = (self.nc, args.resolution_high, args.resolution_wide)

        self.conv1 = layers.Conv2D(6, kernel_size=5, activation="relu", data_format='channels_first') #, input_shape=self.input_shape)
        self.pool = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(16, kernel_size=5, activation="relu", data_format='channels_first')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation="relu")
        self.fc2 = layers.Dense(84, activation="relu")
        self.fc3 = layers.Dense(args.noutputs)

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# Another method of defining a model
# You can also use Sequential mode

# class Net(tfe.Network):
#     """docstring for Net"""
#     def __init__(self, args):
#         super(Net, self).__init__()
#         self.args = args
#         self.nc = args.nchannels
#         self.noutputs = args.noutputs
#         # self.input_shape = (self.nc, args.resolution_high, args.resolution_wide)

#         self.conv1 = self.track_layer(layers.Conv2D(6, kernel_size=5, activation="relu", data_format='channels_first')) #, input_shape=self.input_shape)
#         self.pool = self.track_layer(layers.MaxPool2D(2, 2))
#         self.conv2 = self.track_layer(layers.Conv2D(16, kernel_size=5, activation="relu", data_format='channels_first'))
#         self.flatten = self.track_layer(layers.Flatten())
#         self.fc1 = self.track_layer(layers.Dense(120, activation="relu"))
#         self.fc2 = self.track_layer(layers.Dense(84, activation="relu"))
#         self.fc3 = self.track_layer(layers.Dense(args.noutputs))

#     def call(self, x):
#         x = self.pool(self.conv1(x))
#         x = self.pool(self.conv2(x))
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x

