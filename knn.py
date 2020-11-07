import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random


def bhattacharyya(h1, h2):
    """ Calculates the Byattacharyya distance of two histograms. """

    def normalize(h):
        return h / np.sum(h)

    return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))


def make_knn(X, y):
    clf = KNeighborsClassifier(n_neighbors=1, metric=bhattacharyya)
    clf.fit(X, y)
    return clf


class OpenEndedKNN:

    def first_fit(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=1, metric=bhattacharyya)
        self.model.fit(X, y.ravel())
        self.prev_X, self.prev_y = X, y.ravel()

    def predict(self, input, class_map):
        return self.model.predict(input.reshape(1, -1))[0]

    def add_class_data(self, cls_X, cls_y, class_map):
        self.prev_X = np.append(self.prev_X, cls_X, axis=0)
        self.prev_y = np.append(self.prev_y, cls_y.ravel())
        self.model = KNeighborsClassifier(n_neighbors=1, metric=bhattacharyya)
        self.model.fit(self.prev_X, self.prev_y)

    def current_acc(self, X, y, class_map):
        indices = np.random.choice(len(X), min(len(X), 300))
        return accuracy_score(self.model.predict(X[indices]), y[indices])
