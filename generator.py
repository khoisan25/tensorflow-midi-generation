import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm as tqdm
import midi_manipulation

# 4 Step process


## 1: Hyperparams

lowest_note = midi_manipulation.lowerBound
highest_note = midi_manipulation.upperBound
note_range = highest_note - lowest_note

# Model to use
#     Restricted Boltzmann Machine (RBM): 2 layer nearal net
#         Layer 1 == visible
#         Layer2 == hidden

# Num of timesteps to be created per time
num_timesteps = 15
# Size of layer1, visible layer
n_visible = 2 * note_range * num_timesteps
# Size of layer2, hidden layer
n_hidden = 50

num_epochs = 200
batch_size = 100

# The model's learning rate
lr = tf.constant(0.005, tf.float32)

x = tf.placeholder(tf.float32, [None, n_visible], name="x")
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
bh = tf.Variable(tf.zeros([1, n_hidden], tf-float32, name="bh"))
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))


x_sample = gibbs_sample(1)

h = sample(tf.sigmoid(tf.matmul(x, W) + bh))

h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))



size_bt = tf.cast(tf.shape(x)[0], tf.float32)

W_adder = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h),
    tf.matmul(tf.transpose(x_sample), h_sample)))

bv_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
bh_adder = tf.mul(lr/size_bt. tf.reduce_sum(tf.sub(h, h_sample), 0, True))


updt = [W.assign_add(W_adder), bv.assign_add(bv_adder),
        bh.assign_add(bh_adder)]

# sess = tf.Session()

with tf.Session() as sess:
    init = tf.initialize all variables()
    sess.init()

    for epoch in tqdm (range(num_epochs)):
        for song in songs:
            song = np.array(song)

