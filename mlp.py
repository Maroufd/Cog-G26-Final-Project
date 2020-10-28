from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


def make_mlp(X, y):

    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(y.shape[1], activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    model.fit(X, y, epochs=10, batch_size=16, verbose=1, validation_split=0.2, shuffle=True)
    return model
