__author__ = 'catherinaxu'

import tensorflow as tf
import csv
import os
import textgrid
from tensorflow.contrib import learn
import numpy as np
from tensorflow.contrib import rnn
import sys
import math
import re

np.set_printoptions(threshold=np.nan)

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

labels = np.zeros((len(raw_labels), 2))
for i, label in enumerate(raw_labels):
    labels[i][0] = raw_labels[i]
    labels[i][1] = 1 - raw_labels[i]
model = {}
#got the code to download the glove file in this StackOverflow post:
#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
print ("Loading Glove Model")
f = open("glove.6B.100d.txt",'r')
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = [float(val) for val in splitLine[1:]]
    model[word] = embedding
print ("Done. ",len(model)," words loaded!")

# Build vocabulary
#max_text_length = max([len(x.split(" ")) for x in raw_text])
# Function that maps each email to sequences of word ids. Shorter emails will be padded.
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_text_length)
# x is a matrix where each row contains a vector of integers corresponding to a word.
#x = np.array(list(vocab_processor.fit_transform(raw_text)))
def clean_str(string):
     """
     Tokenization/string cleaning for all datasets except for SST.
     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
     """
     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
     string = re.sub(r"\'s", " \'s", string)
     string = re.sub(r"\'ve", " \'ve", string)
     string = re.sub(r"n\'t", " n\'t", string)
     string = re.sub(r"\'re", " \'re", string)
     string = re.sub(r"\'d", " \'d", string)
     string = re.sub(r"\'ll", " \'ll", string)
     string = re.sub(r",", " , ", string)
     string = re.sub(r"!", " ! ", string)
     string = re.sub(r"\(", " \( ", string)
     string = re.sub(r"\)", " \) ", string)
     string = re.sub(r"\?", " \? ", string)
     string = re.sub(r"\s{2,}", " ", string)
     return string.strip().lower()

stoplist = ["to", "as", "a", "the", "there", "from", "here", "an"]
glove_matrix = []
min = sys.maxint
for line in raw_text:
    count = 0
    add = np.zeros((100,))
    line = clean_str(line).split(" ")

    if len(line) < 50:
        while len(line) < 50:
            line.append("unk")
    for i, word in enumerate(line):
        if i == 50: break
        glove = model.get(word, None)
        if (glove != None and word not in stoplist):
            glove = np.asarray(glove)
            count += 1
            add = np.add(add, glove)
        if (i + 1) % 10 == 0:
            if count == 0:
                glove_matrix.append([0] * 100)
            else:
                average_glove = np.divide(add, count)
                glove_matrix.append(average_glove)

x = np.asarray(glove_matrix)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
#x_shuffled = x[shuffle_indices]
#y_shuffled = labels[shuffle_indices]

train = 0.7
test = 1 - train
# train x, dev x, test x, train y, dev y, test y
train_cutoff = int(0.7 * len(x))
test_cutoff = int(len(x))

y_train_cutoff = int(0.7 * len(labels))
y_test_cutoff = int(len(labels))

train_x = x[0:train_cutoff]
test_x = x[train_cutoff:test_cutoff]
train_y = labels[0:y_train_cutoff]
test_y = labels[y_train_cutoff:y_test_cutoff]


'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 1e-3
training_iters = 100
batch_size = 1239
display_step = 1

# Network Parameters
n_input = 100
n_steps = 5 # timesteps
n_hidden = 50 # hidden layer num of features
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter("tensorboard/", graph=tf.get_default_graph())
    # Keep training until reach max iterations

    while step < training_iters:
        batch_x, batch_y = train_x, train_y
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc, predicted, summary = sess.run([accuracy, pred, merged_summary_op], feed_dict={x: batch_x, y: batch_y})

            summary_writer.add_summary(summary, step)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")

    test_x = test_x.reshape((531, n_steps, n_input))
    print("Testing Accuracy:", \

        sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
