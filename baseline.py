#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import csv
import os
import textgrid
from tensorflow.contrib import learn
import numpy as np

#np.set_printoptions(threshold=np.nan)

#create labels and input features
raw_text = []
raw_labels = []
with open("speeddate/speeddateoutcomes.csv") as outcomes:
    reader = csv.reader(outcomes, delimiter=',', quotechar='|')
    for row in reader:
        txtgrid_path = 'speeddate/' + row[0] + '-' + row[1] + '.TextGrid'
        if os.path.isfile(txtgrid_path):
            if row[10] == '.' or row[14] == '.' or row[14] == '. ' or row[10] == '. ':
                continue
            if (float(row[10]) + float(row[14]))/ 2 >= 5:
                label = 1
            else:
                label = 0
            try:
                txtgrid = textgrid.TextGrid(txtgrid_path)
                txtgrid.read(txtgrid_path)
                female_int_tier = txtgrid.pop()
                male_int_tier = txtgrid.pop()

                # if min time for first interval for female is smaller, they are the first number
                if female_int_tier.indexContaining(0) != None:
                    # ADD THEM
                    data_str = ""
                    for int_tier in female_int_tier:
                        data_str += (int_tier.mark)

                    raw_text.append(data_str)
                elif male_int_tier.indexContaining(0) != None:
                    # ADD THEM
                    data_str = ""
                    for int_tier in male_int_tier:
                        data_str += (int_tier.mark) + " "
                    raw_text.append(data_str)

                raw_labels.append(label)
            except:
            #    print("CONTNUE")
                continue

raw_labels = np.asarray(raw_labels).reshape((len(raw_labels), 1))

# Build vocabulary
max_text_length = max([len(x.split(" ")) for x in raw_text])
# Function that maps each email to sequences of word ids. Shorter emails will be padded.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_text_length)
# x is a matrix where each row contains a vector of integers corresponding to a word.
x = np.array(list(vocab_processor.fit_transform(raw_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(raw_labels)))  # Array of random numbers from 1 to # of labels.
x_shuffled = x[shuffle_indices]
y_shuffled = raw_labels[shuffle_indices]

train = 0.7
test = 0.3
# train x, dev x, test x, train y, dev y, test y
train_cutoff = int(0.7 * len(x_shuffled))
test_cutoff = int(len(x_shuffled))

train_x = x_shuffled[0:train_cutoff]
test_x = x_shuffled[train_cutoff:test_cutoff]
train_y = y_shuffled[0:train_cutoff]
test_y = y_shuffled[train_cutoff:test_cutoff]

# Parameters
learning_rate = 1
training_epochs = 10
batch_size = 1239
display_step = 1

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, max_text_length], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None,1], name='LabelData')

# Set model weights
W = tf.get_variable("Weights", shape=[max_text_length, 1],
           initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.zeros([1]), name='Bias')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
with tf.name_scope('Loss'):
    # Minimize error
    cost = tf.losses.mean_squared_error(y, pred)
    cost = tf.reduce_mean(cost)
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.round(pred), y)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter("tensorboard/", graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_x)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            c, _,  p, labels, summary = sess.run([cost, optimizer, pred, y, merged_summary_op],
                                     feed_dict={x: train_x, y: train_y})
            print(p)
            print(labels)
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: test_x, y: test_y}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


# EXTRA CODE: labels by person
# labels = []
# for i in range(101, 338):
#     sum_so_far = 0
#     count_so_far = 0
#     for j in range(0, len(raw_labels)):
#         if raw_labels[j][0] == i:
#             count_so_far += 2
#             sum_so_far += raw_labels[j][1]
#             sum_so_far += raw_labels[j][2]
#     if count_so_far > 0:
#         avg_rating = float(sum_so_far) / float(count_so_far)
#         if avg_rating >= 5:
#             label = 1
#         else:
#             label = 0
#         labels.append((i, label))
#     else:
#         continue
