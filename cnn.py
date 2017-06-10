#taken from Denny Britz's implementation of Text CNN: https://github.com/dennybritz/cnn-text-classification-tf
import tensorflow as tf
import argparse
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels, load_data_and_labels_thread, load_embedding_vectors_word2vec, load_embedding_vectors_glove
#import yaml
import textgrid
import sys
import nltk
import csv
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
 
# Parameters
# ==================================================
# Model Hyperparameters
learning_rate = 0.00001
training_epochs = 100
batch_size = 1239
display_step = 1
glove_size = 100
filter_sizes = "3,4,5"
 
NUM_NONLEXICAL = 0

def train_tensorflow(x_train, x_test, y_train, y_test):
    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=False,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=100,
                num_classes=2,
                vocab_size=len(x_train),
                embedding_size=100,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=128,
                l2_reg_lambda=0)
 
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
 
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
 
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
 
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
 
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        # out_dir --> runs
        #train_summary_writer = tf.summary.FileWriter("tensorboard_train/", graph=tf.get_default_graph())
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=sess.graph)
 
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # out_dir --> runs
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, graph=sess.graph)
        #dev_summary_writer = tf.summary.FileWriter("tensorboard_dev/", graph=tf.get_default_graph())
 
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
 
        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))
 
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
 
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
 
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
 
    # Generate batches
    train_batches = batch_iter(
        list(zip(x_train, y_train)), batch_size, training_epochs) #batch size, training_epochs
    dev_batches = batch_iter(
        list(zip(x_test, y_test)), batch_size, training_epochs)
    # Training loop. For each batch...
    for train_batch, dev_batch in zip(train_batches, dev_batches):
        x_train_batch, y_train_batch = zip(*train_batch)
        
        train_step(x_train_batch, y_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        print current_step
        if current_step % 100 == 0:
            print("\nEvaluation:")
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
            print("")
        if current_step % 100 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
    
 
 
 
# def generateVinodTest():
 
def main():
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
                except Exception as e:
                    print str(e)
                #    print("CONTNUE")
                    #continue

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


    glove_matrix = []
    vocab_size = 0
    for line in raw_text:
        count = 0
        add = np.zeros((100,))
        for word in line.split(" "):
            glove = model.get(word, None)
            if (glove != None):
                glove = np.asarray(glove)
                count += 1
                add = np.add(add, glove)
        average_glove = np.divide(add, count)
        glove_matrix.append(average_glove)
    x = np.asarray(glove_matrix)
    y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
    x_shuffled = x[shuffle_indices]
    y_shuffled = labels[shuffle_indices]

    train = 0.8
    test = 1 - train
    # train x, dev x, test x, train y, dev y, test y
    train_cutoff = int(0.7 * len(x_shuffled))
    test_cutoff = int(len(x_shuffled))
    train_x = x_shuffled[0:train_cutoff]
    test_x = x_shuffled[train_cutoff:test_cutoff]
    train_y = y_shuffled[0:train_cutoff]
    test_y = y_shuffled[train_cutoff:test_cutoff]
    train_tensorflow(train_x, test_x, train_y, test_y)
 

 
if __name__ == "__main__":
    main()


