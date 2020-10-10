import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def kmeans_display(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]


# label point data with the center nearest
def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return centers, labels, it


def kmeans_with_scikit_learn_module(X, K):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    print('Centers found by scikit-learn:')
    print(kmeans.cluster_centers_)
    pred_label = kmeans.predict(X)
    kmeans_display(X, pred_label)


def kmeans_with_algorithm(X, K):
    (centers, labels, it) = kmeans(X, K)
    print('Centers found by algorithm:')
    print(centers[-1])
    kmeans_display(X, labels[-1])


np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[0.5, 0], [0, 0.5]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

kmeans_with_algorithm(X, K)
