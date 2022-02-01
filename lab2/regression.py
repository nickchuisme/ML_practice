# regression.py
# parsons/2017-2-05
#
# A simple example using regression.
#
# This illustrates both using the linear regression implmentation that is
# built into scikit-learn and the function to create a regression problem.
#
# Code is based on:
#
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#
# Generate a regression problem:
#

# The main parameters of make-regression are the number of samples, the number
# of features (how many dimensions the problem has), and the amount of noise.
X, y = make_regression(n_samples=100, n_features=1, noise = 2)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#
# Solve the problem using the built-in regresson model
#

regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

#
# Evaluate the model
#

# Data on how good the model is:
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plotting training data, test data, and results.
plt.scatter(X_train, y_train, color="black")
plt.scatter(X_test, y_test, color="red")
plt.scatter(X_test, regr.predict(X_test), color="blue")

plt.show()




