import math
from matplotlib.cbook import to_filehandle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



def gradient_descent(X_train, y_train, n_iter=1000, lr=0.0075, tolerance=1e-3):
    w_0, w_1 = 0., 1.
    err_lst = []

    for i in range(n_iter):
        idx = np.random.choice(range(len(X_train)))
        p_i = w_0 + w_1 * X_train[idx]
        err = y_train[idx] - p_i
        err_lst.append(np.abs(err))

        w_0 += lr * err
        w_1 += lr * err * X_train[idx]

        if np.abs(err) < tolerance:
            print(f'Break in {i+1} epoch')
            break

    # plt.plot(err_lst)
    # plt.show()
    return w_0, w_1

def multi_gradient_descent(X_train, y_train, n_iter=1000, lr=0.0075, tolerance=1e-3):
    w_0, w_1, w_2, w_3 = 0., 1., 1., 1.
    err_lst = []

    for i in range(n_iter):
        idx = np.random.choice(range(len(X_train)))
        p_i = w_0 + w_1 * X_train[idx, 0] + w_2 * X_train[idx, 1] + w_3 * X_train[idx, 2]
        err = y_train[idx] - p_i
        err_lst.append(np.abs(err))

        w_0 += lr * err
        w_1 += lr * err * X_train[idx, 0]
        w_2 += lr * err * X_train[idx, 1]
        w_3 += lr * err * X_train[idx, 2]

        if np.abs(err) < tolerance:
            print(f'Break in {i+1} epoch')
            break

    # plt.plot(err_lst)
    # plt.show()
    return w_0, w_1, w_2, w_3

def compute_loss(X_test, y_test, w_0, *args):
    y_hat = w_0
    for i, w in enumerate(args):
        y_hat += w * X_test[:, i]

    return np.mean((y_hat - y_test) ** 2)

if __name__ == "__main__":
    # dataset
    X, y = make_regression(n_samples=100, n_features=3, noise = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    w_0, w_1, w_2, w_3 = multi_gradient_descent(X_train, y_train, lr=0.008)
    loss = compute_loss(X_test, y_test, w_0, w_1, w_2, w_3)
    print("Handcraft Mean squared error: %.2f" % loss)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("Sklearn Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))