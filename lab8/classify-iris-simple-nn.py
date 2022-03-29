# classify-iris-simple-nn.py
# parsons/23-feb-2018
#
# Using a neural network on the iris dataset.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

# Parameters
plot_step = 0.02

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

# Now create a neural network and fit it to the iris data:
iris_nn= MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(5, 2), random_state=1)
iris_nn.fit(X, y)

# Now plot the decision surface that we just learnt by using the neural network to
# classify every packground point.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = iris_nn.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the neural net
                                                   # to predict the classification
                                                   # of each background point.
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# Also plot the original data on the same axes
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))#, cmap='autumn')

# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()
