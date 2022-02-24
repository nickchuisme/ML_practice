# classifier.py
# Ying-Jui Chu/24-Feb-2022
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

"""
I implement Random Forest by programming my algorithm 
about bagging and voting and combining sci-kit learn Decision tree
model. In this coursework, I am looking forward to understanding 
how random forest works and scratching the algorithm. 

Instead of fit() predict() functions, I write bootstrap() and crossvalidation() 
functions to develop random forest and calculate the acuracy regarding my model.
"""
class Classifier:
    """
    The hyperparameters:
    n_trees: 300 decision trees
    criterion: gini index
    max_features: 'sqrt' function
    kfold: 3
    """
    def __init__(self, n_trees=100):
        self.forest = list()
        self.n_trees = n_trees
        self.criterion = 'gini'
        self.max_features = 'sqrt'
        self.kfold = 10

    def reset(self):
        self.forest = list()

    def fit(self, data, target):
        """
        Because I cannot edit the other scripts but only classifier.py.
        I do the trick here when calling this fit function will import
        data to cross_validation function to train model, calculate 
        accuracy and finally return the mean of accuracy.
        """
        accuracy = self.cross_validation(data, target, fold=self.kfold)
        print('Model Trained! Accuracy: {:.4f}'.format(accuracy))

    def predict(self, X, legal=None):
        """
        The central concept here is voting,
        it means the random forest returns one of the
        predictions which have the highest appearance.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(1, len(X))
        elif isinstance(X, np.ndarray):
            X = X.reshape(X.shape[0], X.shape[1])
        # assign all predictions to a list
        predictions = np.array([tree.predict(X) for tree in self.forest])

        final_predictions = []
        for i in range(predictions.shape[1]):
            # sort the list by counting appearances and save the prediction which has a maximum appearance
            count = [(sum(predictions[:, i] == p), p) for p in set(predictions[:, i])]
            max_count = sorted(count, key=lambda x: x[0], reverse=True)[0]
            final_predictions.append(max_count[1])
        # return the predicted number
        return np.array(final_predictions)

    def cross_validation(self, X, y, fold=5):
        """
        Splitting data into training and testing data to do cross-validation 
        due to the small amount of raw data.
        """
        accuracies = []
        X, y = np.array(X), np.array(y)
        for train_idx, test_idx in KFold(n_splits=fold).split(X):
            # reset the random forest in each fold
            self.reset()
            # fit the model with training data
            self.model_train(X[train_idx], y[train_idx])
            # calculate the accuracy of the model with the trained random forest 
            # and a predicted result.
            accuracy = sum(self.predict(X[test_idx]) == y[test_idx]) / X[test_idx].shape[0]
            accuracies.append(accuracy)
        # return the mean of the accuracy of each model
        return np.mean(accuracies)

    def bootstrap(self, data, target):
        """
        Do bootstrap and collect random samples
        """
        idx = np.random.choice(range(len(data)), size=round(len(data) * 0.9), replace=True)
        return np.array(data)[idx], np.array(target)[idx]

    def model_train(self, data, target):
        """
        Here is the function that is doing "fitting" data.
        """
        for i in range(self.n_trees):
            # do bootstrap before fitting
            X, y = self.bootstrap(data, target)
            # fit each decision tree and save it to the forest
            tree_i = DecisionTreeClassifier(criterion=self.criterion, max_features=self.max_features).fit(X, y)
            self.forest.append(tree_i)