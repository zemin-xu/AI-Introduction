

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

import keras
import time
import numpy as np

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes, num_neuron):

    #Application 1 - Step 5 - Initialize the sequential model
    model = Sequential()

    #TODO - Application 1 - Step 5 - build a standard feed-forward network with one dense hidden layer(with 8 neurons) and one dense output layer
    # three hidden layers will be enough for all application
    # num of neuron from 1 to 1024
    model.add(Dense(num_neuron, input_dim=num_pixels, kernel_initializer='normal',
                     activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

     #TODO - Application 1 - Step 6 - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    #TODO - Application 1 - Step 2 - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

    #TODO - Application 1 - Step 3 - Normalize the input values
    X_train = X_train / 255
    X_test = X_test /255

    #TODO - Application 1 - Step 4 - Transform the classes labels into a binary matrix
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    time_callback = TimeHistory()
    times = []
    nums_neuron = [8, 16, 32, 64, 128]
    histories = []
    #Application 1 - Step 5 - Call the baseline_model function
    for i in range(len(nums_neuron)):
        model = baseline_model(num_pixels, num_classes, nums_neuron[i])

    #TODO - Application 1 - Step 7 - Train the model
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              epochs=10, batch_size=100, callbacks=[time_callback], verbose=2)
        for j in range(len(history.epoch)):
            history.epoch[j] = history.epoch[j] + 1
        times.append(time_callback.times)

        histories.append(history)
    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
        scores = model.evaluate(X_test, Y_test, verbose=0)
        #res.append(scores[1])
        #print("baseline error: {:.2f}".format(100 - scores[1] * 100))

    line0, = plt.plot(histories[0].epoch, histories[0].history['accuracy'])
    line1, = plt.plot(histories[1].epoch, histories[1].history['accuracy'])
    line2, = plt.plot(histories[2].epoch, histories[2].history['accuracy'])
    line3, = plt.plot(histories[3].epoch, histories[3].history['accuracy'])
    line4, = plt.plot(histories[4].epoch, histories[4].history['accuracy'])
    plt.legend([line0, line1,line2,line3,line4],
               [nums_neuron[0], nums_neuron[1], nums_neuron[2], nums_neuron[3], nums_neuron[4]])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    for i in range(len(times)):
        mean = np.mean(times[i])
        print("the average time for each epoch in case: ",nums_neuron[i], " of neuron is: ", mean)

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    # Application 2 - Step 6 - Initialize the sequential model
    model = Sequential()

    #TODO - Application 2 - Step 6 - Create the first hidden layer as a convolutional layer



    #TODO - Application 2 - Step 6 - Define the pooling layer



    #TODO - Application 2 - Step 6 - Define the Dropout layer



    #TODO - Application 2 - Step 6 - Define the flatten layer



    #TODO - Application 2 - Step 6 - Define a dense layer of size 128



    #TODO - Application 2 - Step 6 - Define the output layer



    #TODO - Application 2 - Step 7 - Compile the model



    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #TODO - Application 2 - Step 3 - reshape the data to be of size [samples][width][height][channels]



    #TODO - Application 2 - Step 4 - normalize the input values



    #TODO - Application 2 - Step 5 - Transform the classes labels into a binary matrix



    #Application 2 - Step 6 - Call the cnn_model function
    #model = CNN_model((28,28,1), num_classes)


    #TODO - Application 2 - Step 8 - Train the model


    #TODO - Application 2 - Step 8 - System evaluation - compute and display the prediction error


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    #TODO - Application 1 - Step 2 - Train and predict on a MLP
    trainAndPredictMLP(X_train, Y_train,X_test,Y_test)

    #TODO - Application 2 - Train and predict on a CNN
    

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
