import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

plot_step = 0.02
X = bc.data[:, [1, 3]]
y = bc.target

bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

Z = bc_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float64))

plt.xlabel( bc.feature_names[1], fontsize=10 )
plt.ylabel( bc.feature_names[3], fontsize=10 )
plt.show()