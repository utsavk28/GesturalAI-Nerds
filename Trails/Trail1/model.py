from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def load_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.load_weights('./model.h5')
    return model
