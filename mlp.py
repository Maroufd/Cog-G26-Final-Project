import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input


def get_model_body(input_shape):
    """
    Returns the main body of the MLP in this study.

    :param input_shape: number of inputs for the first layer (dimensionality of the feature space)
    :returns: Keras model without classification layer
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(4048, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    return model


def clf_layer(n_outputs):
    """
    :returns: the classification layer based on the number of outputs (categories)
    """
    return Dense(n_outputs, activation="softmax")


def make_mlp(X, y):
    """
    Creates the MLP using the given dataset.

    :param X: the inputs of the dataset
    :param y: the corresponding labels
    :returns: compiled Keras model
    """
    model = get_model_body(X.shape[1:])
    model.add(clf_layer(y.shape[1]))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


class OpenEndedMLP:
    """
    This class implements the MLP/DEN as described in our report
    """

    def __init__(self, min_samples_per_cat=10):
        """
        Initializes the class.

        :param min_samples_per_cat: minimum number of samples per category to use when retraining
        """
        self.model = None
        self.min_samples_per_cat = min_samples_per_cat

    def first_fit(self, X, y, class_map):
        """
        Initial fit with two categories of the model. Trains the MLP for the first time. Remembers
        the dataset.

        :param X: the inputs for the initial dataset
        :param y: the corresponding labels
        :param class_map: dict with for every class in the dataset the position it should have
                          when one hot encoding the labels
        """

        self.X, self.y = X, y
        # One hot encode the labels with the known class mapping (one hot encoding)
        labels = self.transform_labels(y, class_map)
        self.model = make_mlp(X, labels)
        # Fit the model for the first time using terminating conditions
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.98)
        self.model.fit(
            X, labels, epochs=10, batch_size=16, verbose=1, callbacks=[es, tb], shuffle=True
        )

    def add_class_data(self, cls_X, cls_y, class_map):
        """
        Retrains the model when a new category is introduced.

        :param cls_X: inputs for the new category
        :param cls_y: corresponding labels
        :param class_map: updated dict with for every class in the dataset the position it should
                          have when one hot encoding the labels
        """

        # Update the dataset
        self.X = np.append(self.X, cls_X, axis=0)
        self.y = np.append(self.y, cls_y, axis=0)
        # Remember the weights of the last layer
        old_clf_layer_weights, old_clf_layer_bias = self.model.layers[-1].get_weights()

        # Create model with the same weights as the current model, except for the last layer
        new_model = Sequential()
        new_model.add(Input(shape=self.X.shape[1:]))
        for layer in self.model.layers[:-1]:
            new_model.add(layer)

        # Add new classification layer (add one output neuron) and compile the model
        new_model.add(clf_layer(len(class_map)))
        new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Set the learned weights of the old model in the last layer of the new model
        new_clf_layer_weights, new_clf_layer_bias = new_model.layers[-1].get_weights()
        new_clf_layer_weights[:, : old_clf_layer_weights.shape[1]] = old_clf_layer_weights
        new_clf_layer_bias[: old_clf_layer_bias.shape[0]] = old_clf_layer_bias
        new_model.layers[-1].set_weights([new_clf_layer_weights, new_clf_layer_bias])

        # Freeze all weights except those of the last layer
        for layer in new_model.layers[:-1]:
            layer.trainable = False

        # Sample dataset according to the algorithm described in the paper
        # This dataset contains all instances of the new class, and fixed number for old classes
        cls_train_X, cls_train_y = self.get_train_dataset(self.X, self.y, new_cls=cls_y[0, 0])
        labels = self.transform_labels(cls_train_y, class_map)

        # Fit the model until loss stops decreasing or 95% accuracy is reached
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.95)
        new_model.fit(
            cls_train_X,
            labels,
            epochs=50,
            batch_size=16,
            verbose=1,
            callbacks=[es, tb],
            shuffle=True,
        )

        # 'Unfreeze' all layers
        for layer in new_model.layers[:-1]:
            layer.trainable = True

        # Sample dataset where every class is represented equally
        all_train_X, all_train_y = self.get_train_dataset(self.X, self.y)
        labels = self.transform_labels(all_train_y, class_map)

        # Fit the model until loss stops decreasing or 98% accuracy is reached
        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.98)
        new_model.fit(
            all_train_X,
            labels,
            epochs=50,
            batch_size=16,
            verbose=1,
            callbacks=[es, tb],
            shuffle=True,
        )
        self.model = new_model

    def current_acc(self, X, y, class_map):
        """
        Determines the current accuracy of the model given the test dataset. Samples 300 examples
        from the test set randomly.

        :param X: the test inputs
        :param y: the test outputs/lables
        :param class_map: dict with for every class in the dataset the position it should have
                          when one hot encoding the labels
        :returns: current accuracy
        """
        indices = np.random.choice(len(X), min(len(X), 300))
        labels = self.transform_labels(y, class_map)
        return accuracy_score(
            np.argmax(self.model.predict(X[indices]), axis=-1), np.argmax(labels[indices], axis=-1)
        )

    def transform_labels(self, y, class_map):
        """
        Transforms y labels to one hot encoded labels corresponding to the current known number
        of classes. Every class has a fixed position, i.e. the first known class will always have
        position 0 in the one hot encoded vectors, and the last introduced class will always have
        the last position in the one hot encoded vectors. This way, the output neurons will keep
        responding to the same class.

        :param y: the labels to transform
        :param class_map: dict with for every class in the dataset the position it should have
                          when one hot encoding the labels
        :returns: one hot encoded labels
        """
        new_y = np.zeros((y.shape[0], len(class_map)))
        for i, cls in enumerate(y):
            new_y[i, class_map[cls[0]]] = 1.0
        return new_y

    def get_train_dataset(self, X, y, new_cls=None):
        """
        Samples a sub dataset from the given dataset. If a 'new class' is given, includes all
        examples of that new class, and adds the minimum number of samples per category to the
        sub dataset. If no new class is given, samples an equal number of samples for every category
        in the dataset, equal to 50.

        :param X: the inputs to sample from
        :param y: the corresponding labels
        :param new_cls: str, the newly introduced class, can be None
        :returns: subset of X and subset of y
        """

        # First determine the number of instances per class to sample
        if new_cls:
            n_old_classes = np.unique(y).shape[0] - 1
            n_old_class_examples = len(np.where(y == new_cls)[0])
            samples_per_class = max(
                self.min_samples_per_cat, int(n_old_class_examples / n_old_classes)
            )
        else:
            samples_per_class = 50

        train_X, train_y = np.array([]), np.array([])
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            # If a new class is given and this is the new class, add all instances to the dataset
            if new_cls and cls == new_cls:
                new_X = X[cls_indices]
                new_y = y[cls_indices]
            # Else add a randomly sampled subset of defined size of this category to the dataset
            else:
                selection = np.random.choice(cls_indices.shape[0], samples_per_class)
                new_X = X[cls_indices[selection]]
                new_y = y[cls_indices[selection]]

            # If it is the first selected subset
            if train_X.shape[0] == 0:
                train_X = new_X
                train_y = new_y
            # If there is already a subset of data
            else:
                train_X = np.append(train_X, new_X, axis=0)
                train_y = np.append(train_y, new_y, axis=0)
        return train_X, train_y


class TerminateOnBaseline(Callback):
    """
    Taken from:
    https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0

    Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor="accuracy", baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print("\nEpoch %d: Reached baseline, terminating training" % (epoch))
                self.model.stop_training = True
