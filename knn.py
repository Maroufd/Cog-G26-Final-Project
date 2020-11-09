import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def bhattacharyya(h1, h2):
    """
    Calculates the Byattacharyya distance of two histograms.

    :param h1: the first histogram
    :param h2: the second histogram
    :returns: bhattacharyya distance between the two distances
    """

    # Note that the histograms are already normalized by the handcrafted descriptors
    return 1 - np.sum(np.sqrt(np.multiply(h1, h2)))


def symmetricKL(h1, h2):
    """
    Calculates the symmetric kullback-leiber divergence (jensen-shannon) of two histograms.

    Taken from:
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence

    :param h1: the first histogram
    :param h2: the second histogram
    :returns: symmetric kullback-leiber metric of the two distances
    """

    _M = 0.5 * (h1 + h2)
    return 0.5 * (entropy(h1, _M) + entropy(h2, _M))


def make_knn(X, y, distance):
    """Creates a KNN classifier using the given dataset.

    :param X: inputs of the dataset
    :param y: corresponding labels
    :param distance: distance function to use, either "bhattacharyya" or "symmetricKL"
    :returns: sklearn KNN classifier with specified distance function
    """

    distance_f = bhattacharyya if distance == "bhattacharyya" else symmetricKL
    clf = KNeighborsClassifier(n_neighbors=1, metric=distance_f)
    clf.fit(X, y)
    return clf


class OpenEndedKNN:
    """
    This class implements the open ended KNN algorithm
    """

    def first_fit(self, X, y, class_map, distance):
        """Initial fit of the model. Remembers the data

        :param X: inputs of the two initial categories
        :param y: corresponding labels
        :param class_map: not used here, added for generality (used in the MLP approach)
        :param distance: the distance function to use for the KNN algorithm
        """

        self.distance_f = bhattacharyya if distance == "bhattacharyya" else symmetricKL
        self.model = KNeighborsClassifier(n_neighbors=1, metric=self.distance_f)
        self.model.fit(X, y.ravel())
        self.prev_X, self.prev_y = X, y.ravel()

    def add_class_data(self, cls_X, cls_y, class_map):
        """
        Adds class data of a new category to the model. The model is then reinstantiated with this
        new data.

        :param cls_X: inputs of the new category
        :param cls_y: corresponding labels
        :param class_map: not used here, added for generality (used in the MLP approach)
        """
        self.prev_X = np.append(self.prev_X, cls_X, axis=0)
        self.prev_y = np.append(self.prev_y, cls_y.ravel())
        self.model = KNeighborsClassifier(n_neighbors=1, metric=self.distance_f)
        self.model.fit(self.prev_X, self.prev_y)

    def current_acc(self, X, y, class_map):
        """
        Calculates the current accuracy of the model on the given test data. Samples 300 instances
        randomly from the test set

        :param cls_X: test inputs
        :param cls_y: corresponding test labels
        :param class_map: not used here, added for generality (used in the MLP approach)
        :returns: accuracy of the model on the sampled sub test dataset
        """
        indices = np.random.choice(len(X), min(len(X), 300))
        return accuracy_score(self.model.predict(X[indices]), y[indices])
