import collections
import numpy
import csv
import os
import textgrid
import numpy as np
import random
import re



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

def model_stuff(preds_train, feat_vec_train, preds_test, feat_vec_test):
    #clf = RandomForestClassifier()
    clf = LogisticRegression()
    clf.fit(feat_vec_train, preds_train)
    predicted_train = clf.predict(feat_vec_train)
    mean_val_train = np.mean(predicted_train == preds_train)
    tp_train = 0
    tn_train = 0
    fp_train = 0
    fn_train = 0
    for i in range(len(preds_train)):
		if (numpy.array_equal(preds_train[i], predicted_train[i])):
			if (numpy.array_equal(preds_train[i], numpy.asarray([1,0]))):
				tp_train += 1
			else:
				tn_train += 1
		else:
			if (numpy.array_equal(predicted_train[i], numpy.asarray([0,1]))):
				fn_train += 1
			else:
				fp_train += 1


    # print("training accuracy: " + str(mean_val_train))
    # print("training precision: " + str(tp_train * 1.0 / (tp_train + fp_train)))
    # print("training recall: " + str(tp_train * 1.0 / (tp_train + fn_train)))

    predicted_test = clf.predict(feat_vec_test)
    tp_test = 0
    tn_test = 0
    fp_test = 0
    fn_test = 0
    for i in range(len(preds_test)):
		if (numpy.array_equal(preds_test[i], predicted_test[i])):
                        if (numpy.array_equal(preds_test[i], numpy.asarray([0,1]))):
				tp_test += 1
			else:
				tn_test += 1
		else:
                        if (numpy.array_equal(predicted_test[i], numpy.asarray([0,1]))):
				fn_test += 1
			else:
				fp_test += 1
    # print("testing accuracy: " + str(numpy.mean(predicted_test == preds_test)))
    # print("testing precision: " + str(tp_test * 1.0 / (tp_test + fp_test)))
    # print("testing recall: " + str(tp_test * 1.0 / (tp_test + fn_test)))
    return [mean_val_train, tp_train * 1.0 / (tp_train + fp_train), tp_train * 1.0 / (tp_train + fn_train),numpy.mean(predicted_test == preds_test), tp_test * 1.0 / (tp_test + fp_test), tp_test * 1.0 / (tp_test + fn_test)]


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
    for training_ex_feats in lines:
        features_with_index = training_ex_feats.strip().split(" ")
        
        data_mfcc_features.append( [float(x.split(":")[1]) for x in features_with_index]  )
        
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

# CREATING LABELS & INPUT FEATURES
raw_text = []
raw_labels = []
# %%%%%% for mfcc %%%%%%%%
abandon = ["232_219", "219_232", "232_220", "220_232"]
raw_mfcc = []
# %%%%%% end for mfcc %%%%%%%%
with open("../speeddate/speeddateoutcomes.csv") as outcomes:
    reader = csv.reader(outcomes, delimiter=',', quotechar='|')
    for row in reader:
        # %%%%%% for mfcc %%%%%%%%
        possible_filename = str(row[0]) + "_" + str(row[1])
        other_possible_filename = str(row[1]) + "_" + str(row[0])
        # %%%%%% end for mfcc %%%%%%%%
        txtgrid_path = '../speeddate/' + row[0] + '-' + row[1] + '.TextGrid'
        if os.path.isfile(txtgrid_path):
        # %%%%%% for mfcc %%%%%%%%
            if row[10] == '.' or row[14] == '.' or row[14] == '. ' or row[10] == '. ' or possible_filename in abandon:
        # %%%%%% end for mfcc %%%%%%%%
                continue
            if (float(row[10]) + float(row[14]))/2 >= 5:
                label = 1
            else:
                label = 0
            try:
                txtgrid = textgrid.TextGrid(txtgrid_path)
                txtgrid.read(txtgrid_path)
                female_int_tier = txtgrid.pop()
                male_int_tier = txtgrid.pop()    
                
                # whoever has smaller min talked first
                if female_int_tier.indexContaining(0) != None:
                    data_str = ""
                    for int_tier in female_int_tier:
                        data_str += (int_tier.mark) + " "
                        # %%%%%% for mfcc %%%%%%%%
                    if mfcc_file_to_index.get(possible_filename) == None:
                        if mfcc_file_to_index.get(other_possible_filename) != None: 
                            other_index = mfcc_file_to_index[other_possible_filename]
                            raw_mfcc.append( mfcc_features[ other_index ]  )
                    else:
                        raw_mfcc.append( mfcc_features[mfcc_file_to_index[possible_filename]]  )
                        # %%%%%% end for mfcc %%%%%%%%
            
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
            except:
                continue

print(len(raw_mfcc))

# GET DATA UP HERE
all_data = np.asarray(raw_text)
all_labels = np.asarray(raw_labels) #.reshape((len(raw_labels),1))


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
    append_glove = np.append(average_glove, raw_mfcc[i])
    #append_glove = average_glove
    glove_matrix.append(append_glove)

x = np.asarray(glove_matrix)


#---------------------- Randomly shuffle data
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

#------------------------------------


#print train_y
accuracy_train = 0
precision_train = 0
recall_train = 0
accuracy_test = 0
precision_test = 0
recall_test = 0
for i in range(10):
	retval = model_stuff(train_y, train_x, test_y, test_x)
	accuracy_train += retval[0]
	precision_train += retval[1]
	recall_train += retval[2]
	accuracy_test += retval[3]
	precision_test += retval[4]
	recall_test += retval[5]

print "training accuracy: " + str(accuracy_train * 1.0 / 10)
print "training precision: " + str(precision_train * 1.0 / 10)
print "training recall: " + str(recall_train * 1.0 / 10)
print "testing accuracy: " + str(accuracy_test * 1.0 / 10)
print "testing precision: " + str(precision_test * 1.0 / 10)
print "testing recall: " + str(recall_test * 1.0 / 10)






