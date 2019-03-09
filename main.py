import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
import tensorflow as tf
from data_helper import get_market_data, preprocess_data, get_train_validation


BATCH_SIZE = 128
N_EPOCHS = 10


def get_data():
    btc_data = get_market_data("bitcoin")
    train_data, validation_data = get_train_validation(btc_data)
    X_train, y_train = preprocess_data(train_data)
    X_valid, y_valid = preprocess_data(validation_data)

    return X_train, y_train, X_valid, y_valid


def build_model(input_shape):
    model = Sequential()
    model.add(CuDNNLSTM(1024, return_sequences=True, input_shape=(input_shape)))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(1024))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="tanh"))
    
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


def run_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(X_valid, y_valid),
    )


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid = get_data()
    model = build_model(X_train.shape[1:])
    run_model(model, X_train, y_train, X_valid, y_valid)