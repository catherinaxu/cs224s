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
import random
import re
from sklearn.preprocessing import MinMaxScaler


def make_mfcc_map():
    mfcc_files_filepath = "../mfcc_features_filenames.txt" 
    
    open_=open(mfcc_files_filepath,"r")
    lines=open_.readlines();
    
    # maps filename -> index in mfcc_features
    mfcc_map={};

    for i, filename in enumerate(lines):
        filename_itself = filename.strip(".wav\n")
       
        mfcc_map[filename_itself] = i

    return mfcc_map


def read_in_mfcc_features():
    mfcc_filepath = "mfcc_features.txt"
    # We will create an array of arrays
    open_=open(mfcc_filepath,"r")
    lines=open_.readlines()
    data_mfcc_features=[]
    ctr = 0
    for training_ex_feats in lines:
        features_with_index = training_ex_feats.strip().split(" ");
        
        data_mfcc_features.append( [float(x.split(":")[1]) for x in features_with_index]  )
        
        arr = [float(x.split(":")[1]) for x in features_with_index]
        if(len(arr ) != 26):
            print(ctr)
            print("FOUND IT FUCKERSSSS")
            print(len(arr))
            print(arr)
            print(features_with_index)
        ctr += 1
    return data_mfcc_features

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

np.set_printoptions(threshold=np.nan)


mfcc_features = read_in_mfcc_features()
mfcc_file_to_index = make_mfcc_map()


print(mfcc_features[0])
#mfcc_features = MinMaxScaler().fit_transform(mfcc_features)
mfcc_features = MinMaxScaler(feature_range=(0,10)).fit_transform(mfcc_features)

print(mfcc_features[0])

#create labels and input features
raw_text = []
raw_labels = []
raw_mfcc = []
abandon = ["232_219", "219_232", "232_220", "220_232"]
with open("../speeddate/speeddateoutcomes.csv") as outcomes:
    reader = csv.reader(outcomes, delimiter=',', quotechar='|')
    for row in reader:
        possible_filename = str(row[0]) + "_" + str(row[1])
        other_possible_filename = str(row[1]) + "_" + str(row[0])
        txtgrid_path = '../speeddate/' + row[0] + '-' + row[1] + '.TextGrid'
        if os.path.isfile(txtgrid_path):
            if row[10] == '.' or row[14] == '.' or row[14] == '. ' or row[10] == '. ' or possible_filename in abandon:
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

                    if mfcc_file_to_index.get(possible_filename) == None:
                        if mfcc_file_to_index.get(other_possible_filename) != None: 
                            other_index = mfcc_file_to_index[other_possible_filename]
                            raw_mfcc.append( mfcc_features[ other_index ]  )
                    else:
                        raw_mfcc.append( mfcc_features[mfcc_file_to_index[possible_filename]]  )
                    raw_text.append(data_str)
                elif male_int_tier.indexContaining(0) != None:
                    # ADD THEM
                    data_str = ""
                    for int_tier in male_int_tier:
                        data_str += (int_tier.mark) + " "
                    if mfcc_file_to_index.get(possible_filename) == None:
                        if mfcc_file_to_index.get(other_possible_filename) != None: 
                            other_index = mfcc_file_to_index[other_possible_filename]
                            raw_mfcc.append( mfcc_features[ other_index ]  )
                    else:
                        raw_mfcc.append( mfcc_features[mfcc_file_to_index[possible_filename]]  )
                    raw_text.append(data_str)

                raw_labels.append(label)
                #raw_mfcc.append( mfcc_features[mfcc_file_to_index[possible_filename]]  )
                #raw_mfcc.append( mfcc_features[mfcc_map[possible_filename]]  )
            except:
                print("CONTNUE")
                continue
            if mfcc_file_to_index.get(possible_filename) == None:
                if mfcc_file_to_index.get(other_possible_filename) != None: 
                    other_index = mfcc_file_to_index[other_possible_filename]
                    raw_mfcc.append( mfcc_features[ other_index ]  )
                else:
                    print("CONTNUE-2")
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

stoplist = ["to", "as", "a", "the", "there", "from", "here", "an"]

glove_matrix = []
for i, line in enumerate(raw_text):
    line = clean_str(line)
    count = 0
    add = np.zeros((100,))
    for word in line.split(" "):
        glove = model.get(word, None)
        if (glove != None) and word not in stoplist:
            glove = np.asarray(glove)
            count += 1
            add = np.add(add, glove)
    average_glove = np.divide(add, count)
    #append_glove = np.append(average_glove, raw_mfcc[i])
    append_glove = average_glove
    glove_matrix.append(append_glove)

#glove_matrix.pop(0)
#labels = np.delete(labels, 0, axis=0)


# IF USING MFCC FEATURES:
#TODO - FIGURE OUT WHY THIS IS EMPTY
x = np.asarray(glove_matrix)
#x = np.asarray(raw_mfcc)
print(x.shape)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
x_shuffled = x
y_shuffled = labels

train = 0.7
test = 1 - train
# train x, dev x, test x, train y, dev y, test y
train_cutoff = int(0.7 * len(x_shuffled))
test_cutoff = int(len(x_shuffled))

train_x = x_shuffled[0:train_cutoff]
test_x = x_shuffled[train_cutoff:test_cutoff]
train_y = y_shuffled[0:train_cutoff]
test_y = y_shuffled[train_cutoff:test_cutoff]

# Parameters
learning_rate = 1e-3
training_epochs = 100
batch_size = 1239
display_step = 1
glove_size = 100

# IF USING MFCC FEATURES:
#glove_size = 100 + len(mfcc_features[0])


# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, glove_size], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

# check this parameter
HIDDEN_LAYER_SIZE_1 = 50
HIDDEN_LAYER_SIZE_2 = 16
W1 = tf.Variable(tf.random_normal([glove_size, HIDDEN_LAYER_SIZE_1]), name='W1')

# Set model weights
#W1 = tf.get_variable("Weights1", shape=[glove_size, HIDDEN_LAYER_SIZE],
#           initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE_1]), name='b1')

#W2 = tf.get_variable("Weights2", shape=[HIDDEN_LAYER_SIZE, 2],
#           initializer=tf.contrib.layers.xavier_initializer())

W2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2]), name='W2')

#b2 = tf.Variable(tf.zeros([2]), name='Bias2')

b2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE_2]), name='b2')

W3 = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE_2, 2]), name='W3')

b3 = tf.Variable(tf.random_normal([2]), name='b3')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    h = tf.nn.tanh(tf.matmul(x, W1) + b1) # Softmax
    h2 = tf.nn.tanh(tf.matmul(h, W2) + b2)
    pred = tf.matmul(h2, W3) + b3

with tf.name_scope('Loss'):
    # Minimize error
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
    cost = tf.reduce_mean(cost)

with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, "float"))

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

    train_x = np.asarray(train_x)
   # print ("-------TRAIN_X-------")
    #print(train_x)
    #print(train_x.dtype)
    #print(train_x.shape)
    
    train_y = np.asarray(train_y)
    '''
    print ("-------TEST_X INSANCE-------")

    #print(train_x[0])
    #print(test_x[0].dtype)
    #print(test_x[0].shape)
    not_126 = [i for i, feat in enumerate(train_x) if len(feat) != 26]
    actually_126 = [i for i, feat in enumerate(train_x) if len(feat) == 26]
    lens = [len(i) for i in train_x]
    '''

    '''
    print("----NOT 26----")
    print(not_126)
    print("----ACTUALY LENS----")
    print(lens)
    print("at 1096: " + str(len(raw_mfcc[1096])) )
    print("at not_126: " + str(len(raw_mfcc[not_126[0]])) )
    '''

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_x)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            print("len(train_y)=" + str(len(train_y)))
            print("len(train_y[0])=" + str(len(train_y[0])))
            c, _,  p, accuracy, labels, summary = sess.run([cost, optimizer, pred, acc, y, merged_summary_op],
                                     feed_dict={x: train_x, y: train_y})
            # print(labels)
            # print("(labels, predicted_vals)", zip(labels, p))
            # Write logs at every iteration
            print(p)
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)) #, "accuracy=", "{:.9f}".format(accuracy))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    '''
    not_126 = [i for i, feat in enumerate(test_x) if len(feat) != 26]
    actually_126 = [i for i, feat in enumerate(test_x) if len(feat) == 26]
    lens = [len(i) for i in test_x]

    print("----NOT 26----")
    print(not_126)
    print("----ACTUALLY 26----")
    print(len(actually_126))
    print("----ACTUALY LENS----")
    print(lens)

    print("shape")
    print(test_x.shape)
    print(test_y.shape)
'''
    print("Accuracy:", acc.eval({x: test_x, y: test_y}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=tensorboard " \
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
