import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


from confusion_matrix import Confusion_matrix
from cross_validation import loop_cv

if __name__ == "__main__":
    bc = load_breast_cancer()
    result = np.array(loop_cv(clf=KNeighborsClassifier(n_neighbors=5), X=bc.data, y=bc.target, loop=10))
    print(f'socre mean: {result[0].mean()}, std: {result[0].std()}')
    print(f'precision mean: {result[1].mean()}, std: {result[1].std()}')
    print(f'recall mean: {result[2].mean()}, std: {result[2].std()}')