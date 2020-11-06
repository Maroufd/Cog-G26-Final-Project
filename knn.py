import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def bhattacharyya(h1, h2):
    """ Calculates the Byattacharyya distance of two histograms. """

    def normalize(h):
        return h / np.sum(h)

    return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))


def make_knn(X, y):
    clf = KNeighborsClassifier(n_neighbors=1, metric=bhattacharyya)
    clf.fit(X, y)
    return clf
