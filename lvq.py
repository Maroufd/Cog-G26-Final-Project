from neupy import algorithms
import numpy as np
from sklearn.utils import shuffle


def make_lvq(X, y):
    X, y = shuffle(X, y, random_state=69)
    int_y = np.argmax(y, axis=-1)
    lvq = algorithms.LVQ(n_inputs=X.shape[1], n_classes=y.shape[1], verbose=True)
    lvq.train(X, int_y, epochs=100)
    return lvq
