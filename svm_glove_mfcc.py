import collections
import numpy
import csv
import os
import textgrid
import numpy as np

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from scipy.sparse import hstack

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


def make_feature_vectors_train(training, indicies, mfcc):
    vec = TfidfVectorizer(ngram_range=(1,2), max_df=.4)
    data0 = vec.fit_transform(training) # might have issues with size
    #final_data = []
    #for i, elem in enumerate(data):
    #    final_data.append(np.append(elem, raw_mfcc[indicies[i]]))
    data = np.asarray(data0)
    these_feats = mfcc[np.asarray(indicies)]
    mfcc_2 = np.asarray(these_feats)

    print("data: " + str(data.shape))
    print(data)
    print("these_feats: " + str(len(these_feats)))
    print("mfcc_2: " + str(len(mfcc_2)))

    final_data = hstack([data, mfcc_2]).toarray()

    '''
    print "data vector"
    print data
    #print mfcc_2
    print "data"
    print(data.shape)
    print "mfcc_2"
    print(mfcc_2.shape)
    print("^ these should be equa")
#    mfcc = mfcc.reshape((len(mfcc),len(mfcc[0])))
    
    final_data = np.concatenate((data,mfcc_2),axis=1)
    '''
    return (np.asarray(final_data), vec.vocabulary_)

def make_feature_vectors_test(vocab, testing, indicies, mfcc):
    print("make--feature--vectors--test")
    print(mfcc.shape)
    vec = TfidfVectorizer(vocabulary = vocab, ngram_range=(1,2))
    data = vec.fit_transform(testing)
    final_data = []
    for i, elem in enumerate(data):
        final_data.append(np.append(elem, mfcc[indicies[i]]))
    return np.asarray(final_data)

def model_stuff(preds_train, feat_vec_train, preds_test, feat_vec_test):
    clf = RandomForestClassifier()
    clf.fit(feat_vec_train, preds_train)
    predicted_train = clf.predict(feat_vec_train)
    mean_val_train = np.mean(predicted_train == preds_train)
    tp_train = 0
    tn_train = 0
    fp_train = 0
    fn_train = 0
    for i in range(len(preds_train)):
		if (preds_train[i] == predicted_train[i]):
			if (preds_train[i] == 1):
				tp_train += 1
			else:
				tn_train += 1
		else:
			if (predicted_train[i] == 0):
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
		if (preds_test[i] == predicted_test[i]):
			if (preds_test[i] == 1):
				tp_test += 1
			else:
				tn_test += 1
		else:
			if (predicted_test[i] == 0):
				fn_test += 1
			else:
				fp_test += 1
    # print("testing accuracy: " + str(numpy.mean(predicted_test == preds_test)))
    # print("testing precision: " + str(tp_test * 1.0 / (tp_test + fp_test)))
    # print("testing recall: " + str(tp_test * 1.0 / (tp_test + fn_test)))
    return [mean_val_train, tp_train * 1.0 / (tp_train + fp_train), tp_train * 1.0 / (tp_train + fn_train),numpy.mean(predicted_test == preds_test), tp_test * 1.0 / (tp_test + fp_test), tp_test * 1.0 / (tp_test + fn_test)]


raw_mfcc = np.asarray(raw_mfcc)
print(len(raw_mfcc))
indicies = np.arange(len(all_data))
X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(all_data, all_labels, indicies, test_size = 0.3)#, random_state=0)
print(len(indicies))
print("-----train indicies--------")
print(len(indicies_train))
print("-----TEST indicies--------")
print(len(indicies_test))


train_feat_vec, train_vocab = make_feature_vectors_train(X_train, indicies_train, raw_mfcc)
#np.array(list(X_train), dtype=np.float)
print(train_feat_vec.dtype)
train_feat_vec = np.array(train_feat_vec, dtype=np.float)
test_feat_vec = make_feature_vectors_test(train_vocab, X_test, indicies_test, raw_mfcc)

print y_train
accuracy_train = 0
precision_train = 0
recall_train = 0
accuracy_test = 0
precision_test = 0
recall_test = 0
for i in range(10):
	retval = model_stuff(y_train, train_feat_vec, y_test, test_feat_vec)
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






