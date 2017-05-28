import collections
import numpy
import csv
import os
import textgrid
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

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
    print("training accuracy: " + str(mean_val_train))
    predicted_test = clf.predict(feat_vec_test)
    print("testing accuracy: " + str(numpy.mean(predicted_test == preds_test)))

X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size = 0.3)#, random_state=0)

train_feat_vec, train_vocab = make_feature_vectors_train(X_train)
test_feat_vec = make_feature_vectors_test(train_vocab, X_test)

model_stuff(y_train, train_feat_vec, y_test, test_feat_vec)

