# Exercise 2:
# Modify the batch size used to train the model as specified in Table 2. Select a value of 8
# neurons for the hidden layer. What can be observed regarding the system accuracy?

# system performance evaluation
# batch_size        32       64     128     256     512
# sys_accuracy      0.9275  0.9292  0.9249  0.9230  0.9121

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

#####################################################################################################################
#####################################################################################################################


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
              epochs=10, batch_size=512, verbose=2)

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("baseline error: {:.2f}".format(100-scores[1]*100))

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
