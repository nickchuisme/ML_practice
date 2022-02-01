import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from confusion_matrix import Confusion_matrix

def stats_data(prediction, y_test):
    cm = Confusion_matrix()
    cm.calculate_matrix(prediction, y_test)
    cm.update()
    return cm.precision, cm.recall, cm.fscore

def handcraft_cv(clf, X, y, fold=10):
    scores = []
    precision = []
    recall = []

    for train_idx, test_idx in KFold(n_splits=fold).split(X):
        clf = clf.fit(X[train_idx], y[train_idx])
        score = clf.score(X[test_idx], y[test_idx])
        p, r, f1 = stats_data(clf.predict(X[test_idx]), y[test_idx])
        scores.append(score)
        precision.append(p)
        recall.append(r)
    return scores, precision, recall

def loop_cv(clf, X, y, loop=1):
    scores = []
    precision = []
    recall = []

    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
        clf = clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        p, r, f1 = stats_data(clf.predict(X_test), y_test)
        precision.append(p)
        recall.append(r)
    return scores, precision, recall

'''
def cv(clf, X, y, fold=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    clf = clf.fit(X_train, y_train)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    return cv_scores.mean(), cv_scores.std()
'''

if __name__ == "__main__":
    bc = load_breast_cancer()
    result = np.array(handcraft_cv(clf=tree.DecisionTreeClassifier(criterion="entropy"), X=bc.data, y=bc.target, fold=10))
    print(f'socre mean: {result[0].mean()}, std: {result[0].std()}')
    print(f'precision mean: {result[1].mean()}, std: {result[1].std()}')
    print(f'recall mean: {result[2].mean()}, std: {result[2].std()}')