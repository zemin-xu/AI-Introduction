from numpy import mean
from numpy import std
import numpy as np
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from matplotlib import pyplot

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def summarizeLearningCurvesPerformances(histories, scores):

    fig, (ax1, ax2) = pyplot.subplots(2, 1)
    for i in range(len(histories)):
        # plot loss
        #pyplot.subplot(211)
        ax1.set_title('Cross Entropy Loss')
        ax1.plot(histories[i].history['loss'], color='green', label='train')
        ax1.plot(histories[i].history['val_loss'], color='red', label='test')

        # plot accuracy
        #pyplot.subplot(212)
        ax2.set_title('Classification Accuracy')
        ax2.plot(histories[i].history['accuracy'], color='green', label='train')
        ax2.plot(histories[i].history['val_accuracy'], color='red', label='test')

        #print accuracy for each split
        print("Accuracy for set {} = {}".format(i, scores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(mean(scores) * 100, std(scores) * 100, len(scores)))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def prepareData(trainX, trainY, testX, testY):

    #TODO - Application 1 - Step 3 - reshape the data to be of size [samples][width][height][channels]
    trainX= trainX.reshape(trainX.shape[0], 28,   28,   1).astype('float32')
    testX= testX.reshape(testX.shape[0], 28,   28,   1).astype('float32')

    #TODO - Application 1 - Step 4 - normalize the input values
    trainX= trainX/255
    testX= testX/ 255

    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix

    trainY= np_utils.to_categorical(trainY)
    testY= np_utils.to_categorical(testY)

    return trainX, trainY, testX, testY
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):

    # Application 1 - Step 6 - Initialize the sequential model
    model = Sequential()

    #TODO - Application 1 - Step 6 - Create the first hidden layer as a convolutional layer
    model.add(Conv2D(32, (3,3), input_shape=input_shape,
                     activation='relu', kernel_initializer='he_uniform'))

    #TODO - Application 1 - Step 6 - Define the pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    #TODO - Application 1 - Exercise 6 - Add a dropout layer
    # model.add(Dropout(0.2))

    #TODO - Application 1 - Step 6 - Define the flatten layer
    model.add(Flatten(input_shape=input_shape))

    #TODO - Application 1 - Step 6 - Define a dense layer of size 16
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))

    #TODO - Application 1 - Step 6 - Define the output layer
    model.add(Dense(num_classes, activation='softmax'))

    #TODO - Application 1 - Step 6 - Compile the model
    model.compile(SGD(lr=0.01, momentum=0.9) ,loss='categorical_crossentropy',
                  metrics=['accuracy'])



    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndEvaluateClassic(trainX, trainY, testX, testY):

    accuracy = 0

    #Application 1 - Call the defineModel function
    model = defineModel((28, 28, 1), 10)


    #TODO - Application 1 - Step 7 - Train the model
    model.fit(trainX, trainY, epochs = 5, batch_size=32, validation_data=(testX, testY), verbose=1)

    #TODO - Application 1 - Step 7 - Evaluate the model
    loss, accuracy = model.evaluate(testX, testY, verbose=1)
    print("Accuracy = {:.2f}".format(accuracy*100))

    return accuracy
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def trainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5
    scores = []
    histories = []

    #Application 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    #Enumerate splits
    for train_idx, val_idx in kfold.split(trainX):

        #TODO - Application 2 - Step 1 - Select data for train and validation
        kfold = KFold(k_folds, shuffle=True, random_state=1)

        #TODO - Application 2 - Step 1 - Create the model
        trainX_i = trainX[train_idx]
        trainY_i = trainY[train_idx]
        valX_i = trainX[val_idx]
        valY_i = trainY[val_idx]

        model = defineModel((28, 28, 1), 10)

        #TODO - Application 2 - Step 1 - Fit the model
        history = model.fit(trainX_i, trainY_i, epochs=5, batch_size=32, validation_data= (valX_i, valY_i), verbose=1)

        #TODO - Application 2 - Step 1 - Evaluate the model on the test dataset
        loss,accuracy = model.evaluate(testX,testY,verbose=1)
        print("Accuracy={.2f}%".format(accuracy * 100))

        #TODO - Application 2 - Step 1 - Save the accuracy scores in the scores list
        # and the learning history in the histories list
        histories.append(history)
        scores.append(accuracy)

    return scores, histories
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    #TODO - Application 1 - Step 2 - Print the size of the train/test dataset
    print('Train: X={}, Y={}'.format(trainX.shape, trainY.shape))
    print('Test: X={}, Y={}'.format(testX.shape, testY.shape))

    #TODO - Application 1 - Call the prepareData method
    trainX, trainY, testX, testY = prepareData(trainX,trainY,testX,testY)

    #TODO - Application 1 - Step 7 - Train and evaluate the model in the classic way
    #trainAndEvaluateClassic(trainX,trainY, testX, testY)

    #TODO - Application 2 Train and evaluate the model using K-Folds strategy
    scores, histories= trainAndEvaluateKFolds(trainX, trainY, testX,testY)

    #Application 2 - Step2 - System performance presentation
    summarizeLearningCurvesPerformances(histories, scores)

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
