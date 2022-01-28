import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from confusion_matrix import Confusion_matrix

bc = load_breast_cancer()
X = bc.data
y = bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
prediction = knn.predict_proba(X_test)

true_positive_rate = []
false_positive_rate = []
cm = Confusion_matrix()

for threshold in np.arange(0, 1, 0.1):
    new_prediction = np.where(prediction < threshold, 0, 1)
    new_prediction = [p[0] for p in new_prediction]
    cm.calculate_matrix(new_prediction, y_test)
    true_positive_rate.append(cm.true_positive / (cm.true_positive + cm.false_negative))
    false_positive_rate.append(cm.false_positive / (cm.false_positive + cm.true_negative))

plt.scatter(false_positive_rate, true_positive_rate)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve')
plt.show()
