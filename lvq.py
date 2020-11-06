from neupy import algorithms
import datetime as dt
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score


def train_lvq(X, y):
    X, y = shuffle(X, y, random_state=69)
    print(X,y)
    lvqnet = algorithms.LVQ(n_inputs=3888, n_classes=2)
    trained_model = lvqnet.train(X, y, epochs=100)
    lvqnet.fit(X, y, epochs=100, batch_size=8)
    predict=lvqnet.predict(X)
    print("Acc: ", accuracy_score(np.argmax(predict), np.argmax(y)))
    return trained_model


