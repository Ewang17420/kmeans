import random
import numpy as np
import math


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    """
    Implementation of k-means
    :param X:dataset
    :param k:number of clusters
    :param centroids: "kmeans++" or None
    :param max_iter: maximum iteration
    :param tolerance: tolerance of difference between current centroids and previous centroid
    :return: cluster label and final centroids
    """

    m, n = X.shape

    if centroids == 'kmeans++':
        center = select_centroids(X, k)
    else:
        idx = random.sample(range(m), k)
        center = np.array([X[i] for i in idx])

    c_dict = {}
    for i in range(k):
        c_dict[i] = np.empty((0, n))

    count = 0
    new_c = np.zeros((k, n))

    while count < max_iter:

        for x in X:
            cluster_index, distances = cal_shortest_distance(x, center)
            c_dict[cluster_index] = np.vstack((c_dict[cluster_index], x))

        for j in range(k):
            new_c[j] = sum(c_dict[j]) / len(c_dict[j])

        labels = []
        for x in X:
            cluster_index, distances = cal_shortest_distance(x, new_c)
            labels.append(cluster_index)

        if np.all(new_c == center) or np.all(abs(center - new_c) < tolerance):
            break

        center = new_c

        count += 1

    return center, labels


def cal_shortest_distance(x, centroid):
    """
    This is a helper function to calculate the shortest distance among centroids
    :param x: rows in X
    :param centroid: centroid candidates
    :return: the index of centroid and the distance
    """
    shortest = math.inf
    for i in range(len(centroid)):
        dis = sum((centroid[i] - x) ** 2) ** (1 / len(x))
        if dis < shortest:
            shortest = dis
            index = i
    return index, shortest

def select_centroids(X,k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    centers = []
    ind = np.random.choice(range(X.shape[0]), )
    centers.append(X[ind])

    for i in range(k - 1):
        distance = np.sum((np.array(centers) - X[:, None, :])**2, axis=2)

        if i == 0:
            pdf = distance / np.sum(distance)
            centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())]
        else:
            dist_min = np.min(distance, axis=1)
            index_max = np.argmax(dist_min, axis=0)
            centroid_new = X[index_max, :]

        centers.append(centroid_new.tolist())

    return np.array(centers)
