import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def train_AND_model(X_data, y_data):
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_data, y_data, epochs=15000, verbose=2)
    print(model.predict_proba(X_data))

def train_XOR_model(X_data, y_data):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_data, y_data,  epochs=5000, verbose=2)
    print(model.predict_proba(X_data))

def main():

    X_data = np.array([[0,0],[0,1],[1,0],[1,1]], "uint8")
    y_data_AND = np.array([[0],[0],[0],[1]], "uint8")
    y_data_XOR = np.array([[0],[1],[1],[0]], "uint8")

    train_AND_model(X_data, y_data_AND)
    train_XOR_model(X_data, y_data_XOR)

if __name__ == '__main__':
    main()
