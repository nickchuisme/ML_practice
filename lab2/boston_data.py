import math
from matplotlib.cbook import to_filehandle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from my_linear_regression import my_regression, mean_square_error

if __name__ == "__main__":
    boston = load_boston()
    print('running 1 feature gradient descent...')
    X_train, X_test, y_train, y_test = train_test_split(boston.data[:, np.newaxis, 5], boston.target, test_size=0.2, random_state=0)
    model = my_regression(X_train, y_train, lr=0.008)
    mse = mean_square_error(X_test, y_test, model)
    print("Handcraft Mean squared error: %.2f" % mse)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("Sklearn Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))

    ##
    ##
    print('\nrunning 3 feature gradient descent...')
    X_train, X_test, y_train, y_test = train_test_split(boston.data[:, 3:6], boston.target, test_size=0.2, random_state=0)
    w = my_regression(X_train, y_train, lr=0.008)
    mse = mean_square_error(X_test, y_test, w)
    print("Handcraft Mean squared error: %.2f" % mse)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("Sklearn Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))