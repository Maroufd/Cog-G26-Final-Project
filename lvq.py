from neupy import algorithms


def train_lvq(X, y):
    """
    Without testing just the base
    """

    lvqnet = algorithms.LVQ(n_inputs=3870, n_classes=10)
    trained_model = lvqnet.train(X, y, epochs=100)

    return trained_model
