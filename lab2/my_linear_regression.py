import math
from matplotlib.cbook import to_filehandle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def my_regression(X_train, y_train, n_iter=10000, lr=0.0075, tolerance=1e-3):
    X_train = np.c_[np.ones((len(X_train), 1)), X_train]
    w = np.ones([X_train.shape[1], 1])
    err_lst = []

    for i in range(n_iter):
        # lr = 0.01 * np.exp(-0.1*(i+1))
        idx = np.random.choice(range(len(X_train)))
        p_i = np.matmul(X_train[idx], w)
        err = y_train[idx] - p_i
        err_lst.append(np.abs(err))

        # update weights
        w += (lr * err * X_train[idx]).reshape(X_train.shape[1], 1)

        # break iteration if mean error lower than tolerance
        if np.mean(err_lst[-15:]) < tolerance:
            print(f'Break in {i+1} epoch')
            break

    # plt.plot(err_lst)
    # plt.show()
    return w

def mean_square_error(X_test, y_test, w):
    X_test = np.c_[np.ones((len(X_test), 1)), X_test]
    y_hat = np.matmul(X_test, w)
    return np.mean((y_hat - y_test.reshape(y_test.size, 1)) ** 2)


if __name__ == "__main__":
    # dataset
    X, y = make_regression(n_samples=100, n_features=30, noise = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    model = my_regression(X_train, y_train, lr=0.01)
    mse = mean_square_error(X_test, y_test, model)
    print("Handcraft Mean squared error: %.2f" % mse)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("Sklearn Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))

    # compare features between my model and sk-learn model
    my_loss = []
    sk_loss = []
    xtick = range(1, 50, 2)
    for i in xtick:
        X, y = make_regression(n_samples=100, n_features=i, noise = 3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        X_train_ = np.c_[np.ones((len(X_train), 1)), X_train]
        X_test_ = np.c_[np.ones((len(X_test), 1)), X_test]

        model = my_regression(X_train_, y_train, lr=0.01)
        mse = mean_square_error(X_test_, y_test, model)
        my_loss.append(mse)

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        sk_loss.append(np.mean((regr.predict(X_test) - y_test) ** 2))
        print(f'Epoch {i}: myMSE: skMSE = {mse:.2f} : {sk_loss[-1]:.2f}')

    plt.plot(xtick, my_loss, "-*", label="my")
    plt.plot(xtick, sk_loss, "-*", label="sklearn")
    plt.legend()
    plt.show()