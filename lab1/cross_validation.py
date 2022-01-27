import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

bc = load_breast_cancer()
X = bc.data
y = bc.target

def ten_time_cv(loop=10):
    scores = []
    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
        dtree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
        scores.append(dtree.score(X_test, y_test))
    return np.array(scores).mean(), np.array(scores).std()

def cv(fold=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    dtree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    cv_scores = cross_val_score(dtree, X, y, cv=5)
    return cv_scores.mean(), cv_scores.std()

print(ten_time_cv())
print(cv(fold=10))



