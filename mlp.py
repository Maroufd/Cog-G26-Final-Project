import datetime as dt
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


def make_mlp(X, y):
    X, y = shuffle(X, y, random_state=69)

    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Dense(1024, activation="selu"))
    model.add(Dropout(.3))
    model.add(Dense(512, activation="selu"))
    model.add(Dropout(.3))
    model.add(Dense(256, activation="selu"))
    model.add(Dropout(.3))
    model.add(Dense(128, activation="selu"))
    model.add(Dense(y.shape[1], activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)

    log_dir = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.summary()
    model.fit(X, y, epochs=50, batch_size=16, verbose=1, validation_split=0.2, callbacks=[es, tb])
    return model
