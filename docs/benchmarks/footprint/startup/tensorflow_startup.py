# TensorFlow startup-latency benchmark: import tensorflow, construct the same
# small MLP, run one forward pass, print the result, exit.

import tensorflow as tf

net = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1),
])

output = net(tf.ones((1, 10)), training=False)

print("prediction", float(output[0, 0]))
