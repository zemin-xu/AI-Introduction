import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

def AND_model():

    #Application 1 - Step 5 - Initialize the sequential model
    model = Sequential()

    #TODO - Application 1 - Step 5 - build a standard feed-forward network with one dense hidden layer(with 8 neurons) and one dense output layer
    # three hidden layers will be enough for all application
    # num of neuron from 1 to 1024
    # the first para is the neuron numbers
    model.add(Dense(input_dim=2, activation='sigmoid', output_dim=1))
    #model.add(Dense(activation='relu', output_dim=3))
    #model.add(Dense(output_dim =1, activation='sigmoid'))

    #TODO - Application 1 - Step 6 - Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_AND_model(X_data, Y_data):
    model = AND_model()

    #TODO - Application 1 - Step 7 - Train the model
    model.fit(X_data, Y_data, epochs=5000, verbose=2)

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    pred = model.predict(X_data)
    print(pred)

    return

def main():

    X_data = np.array([[0,0],[0,1],[1,0],[0,1]], "uint8")
    Y_data = np.array([[0],[0],[0],[1]], "uint8")

    train_AND_model(X_data, Y_data)

    return

if __name__ == '__main__':
    main()
