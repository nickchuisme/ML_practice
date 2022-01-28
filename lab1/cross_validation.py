import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from confusion_matrix import Confusion_matrix

bc = load_breast_cancer()
X = bc.data
y = bc.target

def stats_data(prediction, y_test):
    cm = Confusion_matrix()
    cm.calculate_matrix(prediction, y_test)
    cm.update()
    return cm.precision, cm.recall, cm.fscore

def ten_time_cv(loop=10):
    scores = []
    precision = []
    recall = []

    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
        dtree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
        scores.append(dtree.score(X_test, y_test))
        p, r, f1 = stats_data(dtree.predict(X_test), y_test)
        precision.append(p)
        recall.append(r)

    return scores, precision, recall

def cv(fold=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    dtree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    cv_scores = cross_val_score(dtree, X, y, cv=5)
    return cv_scores.mean(), cv_scores.std()

if __name__ == "__main__":
    result = np.array(ten_time_cv())
    print(f'socre mean: {result[0].mean()}, std: {result[0].std()}')
    print(f'precision mean: {result[1].mean()}, std: {result[1].std()}')
    print(f'recall mean: {result[2].mean()}, std: {result[2].std()}')

    # print(cv(fold=10))