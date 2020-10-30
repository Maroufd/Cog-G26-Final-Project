import datetime as dt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


def make_mlp(X, y):
    X, y = shuffle(X, y, random_state=69)

    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Dense(512, activation="selu"))
    model.add(Dropout(.5))
    model.add(Dense(256, activation="selu"))
    model.add(Dropout(.5))
    model.add(Dense(128, activation="selu"))
    model.add(Dropout(.5))
    model.add(Dense(64, activation="selu"))
    model.add(Dense(y.shape[1], activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    log_dir = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.summary()
    model.fit(X, y, epochs=100, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es, tb])
    print("Acc: ", accuracy_score(np.argmax(model.predict(X), axis=-1), np.argmax(y, axis=-1)))
    return model
