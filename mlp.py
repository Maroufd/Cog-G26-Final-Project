import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, Callback


def get_model_body(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(4048, activation="relu"))
    model.add(Dropout(.4))
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(.4))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(.4))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(.4))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.4))
    model.add(Dense(128, activation="relu"))
    return model


def clf_layer(n_outputs):
    return Dense(n_outputs, activation="softmax")


def make_mlp(X, y):
    model = get_model_body(X.shape[1:])
    model.add(clf_layer(y.shape[1]))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


class OpenEndedMLP:

    def __init__(self, min_cats=10):
        self.model = None
        self.min_cats = min_cats

    def first_fit(self, X, y, class_map):
        self.X, self.y = X, y
        labels = self.transform_labels(y, class_map)
        self.model = make_mlp(X, labels)
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.98)
        self.model.fit(X, labels, epochs=10, batch_size=16, verbose=1, callbacks=[es, tb], shuffle=True)

    def add_class_data(self, cls_X, cls_y, class_map):
        self.X = np.append(self.X, cls_X, axis=0)
        self.y = np.append(self.y, cls_y, axis=0)
        old_clf_layer_weights, old_clf_layer_bias = self.model.layers[-1].get_weights()

        new_model = Sequential()
        new_model.add(Input(shape=self.X.shape[1:]))
        for layer in self.model.layers[:-1]:
            new_model.add(layer)

        new_model.add(clf_layer(len(class_map)))
        new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        new_clf_layer_weights, new_clf_layer_bias = new_model.layers[-1].get_weights()
        new_clf_layer_weights[:, :old_clf_layer_weights.shape[1]] = old_clf_layer_weights
        new_clf_layer_bias[:old_clf_layer_bias.shape[0]] = old_clf_layer_bias
        new_model.layers[-1].set_weights([new_clf_layer_weights, new_clf_layer_bias])

        # First train the model mostly on the new class, only the last layer is trainable
        for layer in new_model.layers[:-1]:
            layer.trainable = False

        cls_train_X, cls_train_y = self.get_train_dataset(self.X, self.y, new_cls=cls_y[0, 0])
        labels = self.transform_labels(cls_train_y, class_map)

        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.95)
        new_model.fit(cls_train_X, labels, epochs=50, batch_size=16, verbose=1, callbacks=[es, tb], shuffle=True)

        # Next train the model on evenly distributed classes, where all layers are trainable
        for layer in new_model.layers[:-1]:
            layer.trainable = True

        all_train_X, all_train_y = self.get_train_dataset(self.X, self.y)
        labels = self.transform_labels(all_train_y, class_map)

        tb = TerminateOnBaseline(monitor="accuracy", baseline=0.98)
        new_model.fit(all_train_X, labels, epochs=50, batch_size=16, verbose=1, callbacks=[es, tb], shuffle=True)
        self.model = new_model

    def current_acc(self, X, y, class_map):
        indices = np.random.choice(len(X), min(len(X), 300))
        labels = self.transform_labels(y, class_map)
        return accuracy_score(np.argmax(self.model.predict(X[indices]), axis=-1), np.argmax(labels[indices], axis=-1))

    def transform_labels(self, y, class_map):
        """ Transforms y labels to one hot encoded labels corresponding to the current known number
        of classes. """
        new_y = np.zeros((y.shape[0], len(class_map)))
        for i, cls in enumerate(y):
            new_y[i, class_map[cls[0]]] = 1.
        return new_y

    def get_train_dataset(self, X, y, new_cls=None):

        if new_cls:
            n_old_classes = np.unique(y).shape[0] - 1
            n_old_class_examples = len(np.where(y == new_cls)[0])
            samples_per_class = max(self.min_cats, int(n_old_class_examples / n_old_classes))
        else:
            samples_per_class = 50

        train_X, train_y = np.array([]), np.array([])
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            if new_cls and cls == new_cls:
                new_X = X[cls_indices]
                new_y = y[cls_indices]
            else:
                selection = np.random.choice(cls_indices.shape[0], samples_per_class)
                new_X = X[cls_indices[selection]]
                new_y = y[cls_indices[selection]]

            if train_X.shape[0] == 0:
                train_X = new_X
                train_y = new_y
            else:
                train_X = np.append(train_X, new_X, axis=0)
                train_y = np.append(train_y, new_y, axis=0)
        return train_X, train_y


class TerminateOnBaseline(Callback):
    """
    Taken from: https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
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
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
