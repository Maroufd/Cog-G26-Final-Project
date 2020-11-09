import random
import time

import numpy as np
import pandas as pd
from data_utils import extract_X_y
from knn import OpenEndedKNN
from mlp import OpenEndedMLP


def get_random_class_order(y, *, seed):
    """
    Returns a random order of all the available classes.

    :param y: the labels of the dataset
    :param seed: the seed to use to generate the random order
    :returns: list with all unique classes in a random order
    """
    random.seed(seed)
    classes = np.unique(y)
    random.shuffle(classes)
    return classes


def get_examples_from_dataset(X, y, class_to_extract):
    """
    Returns all dataset examples relevant for the given class to extract

    :param X: the object descriptions
    :param y: the corresponding labels
    :param class_to_extract: string, object to extract from the dataset
    :returns: subset of X and y, corresponding to examples of the class to extract
    """
    mask = np.in1d(y, class_to_extract)
    return X[mask], y[mask]


def open_ended_experiment(*, approach, dataset, test_size=0.2):
    """
    Performs an open-ended experimented using the given approach.

    :param approach: the approach to use, must be either "knn" or "mlp"
    :param dataset: the dataset to use, either "ESF_washington" or "GOOD_washington"
    :param test_size: the % of test examples to extract for every category
    """

    # First extract inputs (X) and outputs (y)
    X, y, preprocessor = extract_X_y(dataset)
    original_y = preprocessor.inverse_transform(y)
    # Determine a random class order
    class_order = get_random_class_order(original_y, seed=42)

    # Instantiate an open ended model, either KNN-based or MLP-based (DEN)
    if type == "knn":
        model = OpenEndedKNN()
    else:
        model = OpenEndedMLP()

    # Extract data for the first two categories
    first_X, first_y = get_examples_from_dataset(X, original_y, class_order[0])
    second_X, second_y = get_examples_from_dataset(X, original_y, class_order[1])
    class_map = {class_order[0]: 0, class_order[1]: 1}

    # Extract test and train data for the first two categories
    first_split = int(len(first_X) * test_size)
    second_split = int(len(second_X) * test_size)

    init_X = np.append(first_X[first_split:], second_X[second_split:], axis=0)
    init_y = np.append(first_y[first_split:], second_y[second_split:], axis=0)
    model.first_fit(init_X, init_y, class_map)

    test_X, test_y = np.array([]), np.array([])
    test_X = np.append(first_X[:first_split], second_X[:second_split], axis=0)
    test_y = np.append(first_y[:first_split], second_y[:second_split], axis=0)

    results = []
    # Incrementally introduce new classes, starting at class/category 3
    for n_classes in range(2, class_order.shape[0]):
        train_start = time.time()
        # Extract the new class
        new_class = class_order[n_classes]

        # Give it an unique position when one hot encoding the classes (used for the DEN/MLP)
        class_map[new_class] = n_classes

        # Extract the data for this category
        cls_X, cls_y = get_examples_from_dataset(X, original_y, new_class)

        # Extract test data for this category
        split = int(len(cls_X) * test_size)
        test_X = np.append(test_X, cls_X[:split], axis=0)
        test_y = np.append(test_y, cls_y[:split], axis=0)

        # Retrain the model with the new added data
        model.add_class_data(cls_X[split:], cls_y[split:], class_map)
        train_end = time.time()
        clf_start = time.time()

        # Get current accuracy of the model
        acc = model.current_acc(test_X, test_y, class_map)
        clf_end = time.time()
        print(
            "Acc with {} classes: {}, classification: {} sec, (re)training: {}".format(
                n_classes,
                acc,
                clf_end - clf_start,
                train_end - train_start,
            )
        )
        results.append(
            {
                "accuracy": acc,
                "n_classes": n_classes,
                "clf_time": clf_end - clf_start,
                "train_time": train_end - train_start,
            }
        )

    pd.DataFrame(results).to_csv(f"results/results_open_ended_{type}_{dataset}.csv", index=False)


if __name__ == "__main__":
    open_ended_experiment(approach="mlp", dataset="ESF_washington", test_size=0.2)
