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

# CREATING LABELS & INPUT FEATURES
raw_text = []
raw_labels = []
with open("speeddate/speeddateoutcomes.csv") as outcomes:
    reader = csv.reader(outcomes, delimiter=',', quotechar='|')
    for row in reader:
        txtgrid_path = 'speeddate/' + row[0] + '-' + row[1] + '.TextGrid'
        if os.path.isfile(txtgrid_path):
            if str.strip(row[10]) == '.' or str.strip(row[14]) == '.':
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
                    
                    raw_text.append(data_str)
                else:
                    data_str = ""
                    for int_tier in male_int_tier:
                        data_str += (int_tier.mark) + " "
                    raw_text.append(data_str)
            
                raw_labels.append(label)
            except:
                continue

# GET DATA UP HERE

all_data = np.asarray(raw_text)
all_labels = np.asarray(raw_labels) #.reshape((len(raw_labels),1))


def make_feature_vectors_train(training):
    vec = TfidfVectorizer(ngram_range=(1,2), max_df=.4)
    data = vec.fit_transform(training) # might have issues with size
    return (data, vec.vocabulary_)

def make_feature_vectors_test(vocab, testing):
    vec = TfidfVectorizer(vocabulary = vocab, ngram_range=(1,2))
    data = vec.fit_transform(testing)
    return data

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

X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size = 0.3)#, random_state=0)

train_feat_vec, train_vocab = make_feature_vectors_train(X_train)
test_feat_vec = make_feature_vectors_test(train_vocab, X_test)

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






