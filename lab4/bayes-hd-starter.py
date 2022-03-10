# bayes-hd-starter.py
# parsons/25-feb-2017
#
# The input data is the processed Cleveland data from the "Heart
# Diesease" dataset at the UCI Machine Learning repository:
#
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#
# Define constants
#

datafilename = 'processed.cleveland.data' # input filename
age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file. 
num_features = 13

def load_csv(datafilename):
    data, target = [], []
    with open(datafilename) as infile:
            for row in infile.readlines():
                row = row.strip().split(',')
                if '?' in row:
                    continue
                data.append(row[:-1])
                target.append(row[-1])
    return np.array(data, dtype=float), np.array(target, dtype=int)

def naive_bayes(X_train, X_test, y_train, y_test, data_idx=[1, 2, 5, 6, 8, 10, 11]):

    y_sample = list(set(y_train))
    y_num = np.array([sum(y_train==i) for i in range(len(y_sample))])
    priors = y_num / len(y_train)

    sex_sample = list(set(X_train[:, 1]))
    sex = np.zeros((len(priors), len(sex_sample)), dtype=float)
    cp_sample = list(set(X_train[:, 2]))
    cp = np.zeros((len(priors), len(cp_sample)), dtype=float)
    fbs_sample = list(set(X_train[:, 5]))
    fbs = np.zeros((len(priors), len(fbs_sample)), dtype=float)
    restecg_sample = list(set(X_train[:, 6]))
    restecg = np.zeros((len(priors), len(restecg_sample)), dtype=float)
    exang_sample = list(set(X_train[:, 8]))
    exang = np.zeros((len(priors), len(exang_sample)), dtype=float)
    slope_sample = list(set(X_train[:, 10]))
    slope = np.zeros((len(priors), len(slope_sample)), dtype=float)
    ca_sample = list(set(X_train[:, 11]))
    ca = np.zeros((len(priors), len(ca_sample)), dtype=float)

    # for y in range(len(set(target))):
    for row, y in zip(X_train, y_train):
        sex[y_sample.index(y)][sex_sample.index(row[1])] += 1
        cp[y_sample.index(y)][cp_sample.index(row[2])] += 1
        fbs[y_sample.index(y)][fbs_sample.index(row[5])] += 1
        restecg[y_sample.index(y)][restecg_sample.index(row[6])] += 1
        exang[y_sample.index(y)][exang_sample.index(row[8])] += 1
        slope[y_sample.index(y)][slope_sample.index(row[10])] += 1
        ca[y_sample.index(y)][ca_sample.index(row[11])] += 1


    sex = sex / y_num.reshape(len(y_num), 1)
    cp = cp / y_num.reshape(len(y_num), 1)
    fbs = fbs / y_num.reshape(len(y_num), 1)
    restecg = restecg / y_num.reshape(len(y_num), 1)
    exang = exang / y_num.reshape(len(y_num), 1)
    slope = slope / y_num.reshape(len(y_num), 1)
    ca = ca / y_num.reshape(len(y_num), 1)

    y_pred   = 0
    score    = 0
    
    for test_x, test_y in zip(X_test, y_test):
        prob = np.array([1] * 5, dtype=float)
        prob *= sex[:, sex_sample.index(test_x[1])]
        prob *= cp[:, cp_sample.index(test_x[2])]
        prob *= fbs[:, fbs_sample.index(test_x[5])]
        prob *= restecg[:, restecg_sample.index(test_x[6])]
        prob *= exang[:, exang_sample.index(test_x[8])]
        prob *= slope[:, slope_sample.index(test_x[10])]
        prob *= ca[:, ca_sample.index(test_x[11])]
        prob *= priors

        y_pred = np.where(prob == max(prob))[0].astype(int)
        # if test_y == y_pred:
        #     score += 1
        if test_y == 0 and y_pred == 0:
            score += 1
        elif test_y != 0 and y_pred != 0:
            score += 1
    return score

def sk_naive_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return sum(clf.predict(X_test) == y_test)

if __name__ == "__main__":
    #
    # Open and read data file in csv format
    #
    # After processing:
    # 
    # data   is the variable holding the features;
    # target is the variable holding the class labels.
    data, target = load_csv(datafilename)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=3)

    # decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    print(f"Decision tree score: {clf.score(X_test, y_test)}")

    # naive bayes by scratching
    score = naive_bayes(X_train, X_test, y_train, y_test)
    print(f"Naive Bayes score: {score}/{len(y_test)} = {score/len(y_test)}")
    # sci-kit learn naive bayes
    sk_score = sk_naive_bayes(X_train, X_test, y_train, y_test)
    print(f"Naive Bayes(Sk learn) score: {sk_score}/{len(y_test)} = {sk_score/len(y_test)}")