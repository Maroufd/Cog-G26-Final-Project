import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def extract_X_y(csv_name):
    """
    Reads in csv data, returns inputs and labels one hot encoded and the preprocessor.

    :param csv_name: the name of csv file to load (without extension)
    :returns: inputs, one hot encoded labels and the corresponding preprocessor
    """

    hists = pd.read_csv(f"data/{csv_name}.csv", header=None, index_col=False)

    inputs = hists.iloc[:, :-1].astype(float).to_numpy()
    # Extract labels as strings, i.e. "Vase", "Bottle", etc..
    labels = hists.iloc[:, -1]
    labels = labels.apply(lambda x: x.split("/")[-1].split("_")[0])
    labels = labels.to_numpy().reshape(-1, 1)

    preprocessor = OneHotEncoder(sparse=False)
    labels_onehot = preprocessor.fit_transform(labels)

    return inputs, labels_onehot, preprocessor
