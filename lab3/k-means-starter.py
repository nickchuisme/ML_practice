# k-means-starter.py
# parsons/28-feb-2017
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

#
# Put your K-means code here.
#
def kmean(X, y, center_num=3, iter=100, dirty_random=False):
    # select random centroids
    for _ in range(iter):
        rand_idx = np.random.choice(len(X), size=center_num, replace=True)
        if dirty_random and len(set(y[rand_idx])) != center_num:
            pass
        else:
            centroids = X[rand_idx, :]
            centroids_path = [centroids]
            break

    for i in range(iter):
        # clustering
        labels = []
        clusters = [[] for i in range(center_num)]
        for x, y_ in zip(X, y):
            dis_lst = calc_distance(x, centroids)
            clusters[np.argmin(dis_lst)].append(x)
            labels.append(np.argmin(dis_lst))
        # update centroids
        centroids = [np.mean(c, axis=0) for c in np.array(clusters, dtype=object)]
        centroids_path.append(centroids)
        # print([len(c) for c in clusters])
        if (np.equal(*centroids_path[-2:])).all():
            break
    return np.array(centroids).reshape(center_num, X.shape[1]), np.array(centroids_path), labels

def calc_distance(x, centroids):
    return [(sum((x - i) ** 2)) ** 0.5 for i in centroids] 


if __name__ == "__main__":
    # Load the data
    iris = load_iris()
    X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
    y = iris.target

    x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()

    centroids, test, labels = kmean(X, y, center_num=len(set(y)))
    print(f'Adjusted Rand Index: {adjusted_rand_score(labels, y)}')
    # scikit-learn library
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    sk_centroids = kmeans.cluster_centers_
    #
    # Plot everything
    #
    plt.subplot( 1, 2, 1 )
    # Plot the original data 
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(float), alpha=0.5)
    for i in range(len(test)):
        plt.scatter(test[i][:, 0], test[i][:, 1], c='blue', s=120, marker='x', alpha=0.4)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=330, marker='+', label="Final")
    # Label axes
    plt.xlabel( iris.feature_names[1], fontsize=10 )
    plt.ylabel( iris.feature_names[3], fontsize=10 )
    plt.legend()

    plt.subplot( 1, 2, 2 )
    # Plot the scikit-learn data 
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(float), alpha=0.5)
    plt.scatter(sk_centroids[:, 0], sk_centroids[:, 1], c='black', s=330, marker='+', label="Final")
    plt.show()
