from collections import defaultdict
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from data_utils import extract_X_y
from mlp import make_mlp
from lvq import train_lvq
from knn import make_knn

if __name__ == "__main__":
    X, y, preprocessor = extract_X_y("GOOD_washington")

    strat_split = StratifiedKFold(n_splits=10, test_size=0.1, random_state=42)

    results = defaultdict(list)
    for train_idx, test_idx in strat_split.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        knn = make_knn(X_train, y_train)
        mlp = make_mlp(X_train, y_train)
        # lvqnet = train_lvq(X,y)

        results["knn"].append(accuracy_score(np.argmax(knn.predict(X_test), axis=-1), np.argmax(y_test, axis=-1)))
        results["mlp"].append(accuracy_score(np.argmax(mlp.predict(X_test), axis=-1), np.argmax(y_test, axis=-1)))
