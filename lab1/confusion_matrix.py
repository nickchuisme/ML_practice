import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


class Confusion_matrix:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def calculate_matrix(self, prediction, y_test):
        for p, y in zip(prediction, y_test):
            if p and y:
                self.true_positive += 1
            elif p and not y:
                self.false_positive += 1
            elif not p and y:
                self.false_negative += 1
            else:
                self.true_negative += 1
    
    def update(self, beta=1):
        self.precision = self.true_positive / (self.true_positive + self.false_positive)
        self.recall = self.true_positive / (self.true_positive + self.false_negative)
        self.fscore = self.f_score(beta)

    def f_score(self, beta=1):
        return (1 + beta ** 2) * (self.precision * self.recall) / (beta * beta * self.precision + self.recall)

if __name__ == "__main__":

    bc = load_breast_cancer()
    X = bc.data
    y = bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
    yr_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    prediction = yr_tree.predict(X_test)

    cm = Confusion_matrix()
    cm.calculate_matrix(prediction, y_test)
    cm.update()

    print(f'precision: {cm.precision}')
    print(f'recall: {cm.recall}')
    print(f'f1-score: {cm.fscore}')


