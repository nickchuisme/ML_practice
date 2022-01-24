import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

plot_step = 0.02
X = bc[:, [1, 3]]
Y = bc.target
print(bc)