import numpy as np
from neupy import algorithms
from sklearn.utils import shuffle


def make_lvq(X, y):
    """
    Creates an LVQ network and trains it according to the test data.

    :param X: the inputs to train on
    :param y: the corresponding labels
    :returns: trained lvq network
    """
    # First randomly shuffle the dataset, to mix up class order
    X, y = shuffle(X, y, random_state=69)

    # Transform the labels to sparse labels
    int_y = np.argmax(y, axis=-1)

    # Construct and train the lvq net
    lvq = algorithms.LVQ(n_inputs=X.shape[1], n_classes=y.shape[1], verbose=True)
    lvq.train(X, int_y, epochs=100)
    return lvq
