import numpy as np
import random
import time
import pandas as pd

from data_utils import extract_X_y
from knn import OpenEndedKNN
from mlp import OpenEndedMLP


def get_random_class_order(y, *, seed):
    """
    Returns a random order of all the available classes.
    """
    random.seed(seed)
    classes = np.unique(y)
    random.shuffle(classes)
    return classes


def get_examples_from_dataset(X, y, class_to_extract):
    """ Returns all dataset examples relevant for the given class to extract. """
    mask = np.in1d(y, class_to_extract)
    return X[mask], y[mask]


def add_old_cats(dataset, old_classes, min_cats, cls_X, cls_y):
    new_X, new_y = np.copy(cls_X), np.copy(cls_y)
    for cls in old_classes:
        cls_X, cls_y = dataset.sample(cls, min_cats)
        new_X = np.append(new_X, cls_X, axis=0)
        new_y = np.append(new_y, cls_y, axis=0)
    return new_X, new_y


def open_ended_experiment():
    type = "knn"
    dataset = "ESF_washington"
    test_size = .2
    X, y, preprocessor = extract_X_y(dataset)
    original_y = preprocessor.inverse_transform(y)
    class_order = get_random_class_order(original_y, seed=42)

    if type == "knn":
        model = OpenEndedKNN()
    else:
        model = OpenEndedMLP()

    first_X, first_y = get_examples_from_dataset(X, original_y, class_order[0])
    second_X, second_y = get_examples_from_dataset(X, original_y, class_order[1])

    class_map = {class_order[0]: 0, class_order[1]: 1}

    split = int(len(first_X) * test_size)
    first_X = first_X[split:]
    first_y = first_y[split:]

    split = int(len(second_X) * test_size)
    second_X = second_X[split:]
    second_y = second_y[split:]

    init_X = np.append(first_X, second_X, axis=0)
    init_y = np.append(first_y, second_y, axis=0)
    model.first_fit(init_X, init_y, class_map)

    test_X, test_y = np.array([]), np.array([])
    test_X = np.append(first_X[:split], second_X[:split], axis=0)
    test_y = np.append(first_y[:split], second_y[:split], axis=0)

    results = []
    for n_classes in range(2, class_order.shape[0]):
        train_start = time.time()
        new_class = class_order[n_classes]
        class_map[new_class] = n_classes
        cls_X, cls_y = get_examples_from_dataset(X, original_y, new_class)

        split = int(len(cls_X) * test_size)
        test_X = np.append(test_X, cls_X[:split], axis=0)
        test_y = np.append(test_y, cls_y[:split], axis=0)

        model.add_class_data(cls_X[split:], cls_y[split:], class_map)
        train_end = time.time()
        clf_start = time.time()
        acc = model.current_acc(test_X, test_y, class_map)
        clf_end = time.time()
        print(f"Acc with {n_classes} classes: {acc}, classification: {clf_end - clf_start} sec, (re)training: {train_end - train_start}")
        results.append({"accuracy": acc, "n_classes": n_classes, "clf_time": clf_end - clf_start, "train_time": train_end - train_start})

    pd.DataFrame(results).to_csv(f"results/results_open_ended_{type}_{dataset}.csv", index=False)


if __name__ == "__main__":
    open_ended_experiment()
