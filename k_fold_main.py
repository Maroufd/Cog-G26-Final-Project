import numpy as np
import pandas as pd
from data_utils import extract_X_y
from knn import make_knn
from lvq import make_lvq
from mlp import make_mlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == "__main__":

    dataset = "GOOD_restaurant"
    # Extract inputs and outputs from the dataset
    X, y, preprocessor = extract_X_y(dataset)

    # Create K fold cross validation splits with where the distribution of the labels are preserved
    strat_split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)

    results = []
    # For every split train all three classifiers and remember their performance
    for i, (train_idx, test_idx) in enumerate(strat_split.split(X, np.argmax(y, axis=-1))):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        y_classes = np.argmax(y_test, axis=-1)

        knn = make_knn(X_train, y_train)
        mlp = make_mlp(X_train, y_train)
        lvq = make_lvq(X_train, y_train)

        results.append(
            {
                "run": i,
                "model": "knn",
                "score": accuracy_score(np.argmax(knn.predict(X_test), axis=-1), y_classes),
            }
        )
        results.append(
            {
                "run": i,
                "model": "mlp",
                "score": accuracy_score(np.argmax(mlp.predict(X_test), axis=-1), y_classes),
            }
        )
        results.append(
            {"run": i, "model": "lvq", "score": accuracy_score(lvq.predict(X_test), y_classes)}
        )

    pd.DataFrame(results).to_csv(f"results/k_fold_results_{dataset}.csv")
