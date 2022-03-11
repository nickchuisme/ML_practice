import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

def load_csv(datafilename, column=True):
    data, target = [], []
    with open(datafilename) as infile:
        if column:
            rows = infile.readlines()[1:]
        else:
            rows = infile.readlines()

        for row in rows:
            row = row.strip().split(',')
            if '?' in row:
                continue
            data.append(row[:-1])
            target.append(row[-1])
    return np.array(data, dtype=float), np.array(target, dtype=int)

def sk_svc(data, target):
    # plot background
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel, ax in zip(kernels, axes.ravel()):
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        cs = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Paired)
        ax.scatter(data[:, 0], data[:, 1], c=target.astype(float), s=1)
        ax.set_title(kernel)
    plt.tight_layout()
    plt.show()

def sk_svr(data, target):
    data = data[:, [0, 1]]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel, ax in zip(kernels, axes.ravel()):
        clf = svm.SVR(kernel=kernel, C=50, epsilon=1)
        clf.fit(X_train, y_train)
        mse = np.mean((clf.predict(X_test) - y_test) ** 2)
        print(f'Kernel {kernel}\'s MSE: {mse}')
        # print(X_train[clf.support_].shape[0], y_train[clf.support_].ravel())
        # ax.scatter(X_train[clf.support_][:, 1], y_train[clf.support_].ravel(), color="black")
        # ax.scatter(X_test[np.setdiff1d(np.arange(len(X_test)), clf.support_)], y_test[np.setdiff1d(np.arange(len(X_test)), clf.support_)], color="red")
        # ax.scatter(X_test[:,0], clf.predict(X_test), color="blue", s=1)

    # plt.show()

if __name__ == "__main__":
    BANKNOTE_PATH = './data_banknote_authentication.txt'
    data, target = load_csv(BANKNOTE_PATH, column=False)
    data = data[:, [0, 2]]
    sk_svc(data, target)

    WINE_PATH = './winequality-white.csv'
    data, target = load_csv(WINE_PATH, column=True)
    sk_svr(data, target)
