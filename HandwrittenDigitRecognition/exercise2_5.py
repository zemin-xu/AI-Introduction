# Exercise 5:
# Write a novel Python script that loads the saved weights (model.load_weights) at Exercise
# 4 and make a prediction on the first 5 images of the testing dataset (mnist.test_images()).

# Hint: First we have to specify the artificial neural network architecture as in Step 5 (Application 1), then
# load the set of weights (saved at Exercise 4) and make the prediction for the inputs using the predict()
# function. Keep in mind that the output of our network is the probability of that image to belong to the
# ten classes considered (because of softmax). So, it is necessary to use np.argmax() to return the class
# with the maximum probability score.

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np

#####################################################################################################################
#####################################################################################################################


def load_and_test(X_test, Y_test):

    # TODO - Application 1 - Step 2 - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = X_test.shape[1] * X_test.shape[2]
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

    # TODO - Application 1 - Step 3 - Normalize the input values
    X_test = X_test / 255

    # TODO - Application 1 - Step 4 - Transform the classes labels into a binary matrix
    Y_test_categ = np_utils.to_categorical(Y_test)
    num_classes = Y_test_categ.shape[1]

    # Application 1 - Step 5 - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)

    model.load_weights('weights.h5')

    temp = model.predict(X_test)
    Y_pred = np.argmax(temp, axis = 1)
    print("prediction: ",  Y_pred[:5])
    print("actual: ", Y_test[:5])

#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):

    #Application 1 - Step 5 - Initialize the sequential model
    model = Sequential()

    #TODO - Application 1 - Step 5 - build a standard feed-forward network with one dense hidden layer(with 8 neurons) and one dense output layer
    # three hidden layers will be enough for all application
    # num of neuron from 1 to 1024
    model.add(Dense(8, input_dim=num_pixels, kernel_initializer='normal',
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

    #Application 1 - Step 5 - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)

    #TODO - Application 1 - Step 7 - Train the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              epochs=10, batch_size=200, verbose=2)

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("baseline error: {:.2f}".format(100-scores[1]*100))

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    #X_test = X_test[:5]
    #Y_test = Y_test[:5]
    load_and_test(X_test,Y_test)



    #TODO - Application 1 - Step 2 - Train and predict on a MLP

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
