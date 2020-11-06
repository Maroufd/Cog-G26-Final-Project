from collections import defaultdict
import numpy as np
import pprint
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from data_utils import extract_X_y
from mlp import make_mlp
from lvq import make_lvq
from knn import make_knn

if __name__ == "__main__":
    X, y, preprocessor = extract_X_y("GOOD_washington")

    strat_split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)

    results = defaultdict(list)
    for train_idx, test_idx in strat_split.split(X, np.argmax(y, axis=-1)):
        start = time.time()
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        y_classes = np.argmax(y_test, axis=-1)

        print("training knn")
        knn = make_knn(X_train, y_train)
        print("training mlp")
        mlp = make_mlp(X_train, y_train)
        print("training lvq")
        lvq = make_lvq(X_train, y_train)

        results["knn"].append(accuracy_score(np.argmax(knn.predict(X_test), axis=-1), y_classes))
        results["mlp"].append(accuracy_score(np.argmax(mlp.predict(X_test), axis=-1), y_classes))
        results["lvq"].append(accuracy_score(lvq.predict(X_test), np.argmax(y_test, axis=-1)))

        pprint.pprint(results)
        end = time.time()
        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()
