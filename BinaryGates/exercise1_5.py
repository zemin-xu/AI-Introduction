# compare to sigmoid method, the tanh way learns faster to get the same precision
# tanh: about 250 steps
# sigmoid: about 6000 steps


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(n):

    # Define the sigmoid function as activation function
    return 1.0/(1.0 + np.exp(-n))

def sigmoidDerivative(n):

    return n*(1-n)

def tanh(u):
    return 2 / (1 + np.exp((-2) * u)) - 1

def tanhDerivative(u):
    return (1+u) * (1 - u)


def forwardPropagationLayer(p, weights, biases):

    a = None  # the layer output

    # Multiply weights with the input vector (p) and add the bias   =>  n
    n = np.dot(p, weights) + biases

    # Pass the result to the activation function  =>  a
    a = tanh(n)

    return a

def main():

    #Application 2 - Train a ANN in order to predict the output of an XOR gate.
    #The network should receive as input two values (0 or 1) and should predict the target output

    #Input data
    points = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    #Labels
    labels = np.array([[0], [1], [1], [0]])

    # Initialize the weights and biases with random values
    inputSize = 2
    noNeuronsLayer1 = 2
    noNeuronsLayer2 = 1

    weightsLayer1 = np.random.uniform(size=(inputSize, noNeuronsLayer1))
    weightsLayer2 = np.random.uniform(size=(noNeuronsLayer1, noNeuronsLayer2))

    biasLayer1 = np.random.uniform(size=(1, noNeuronsLayer1))
    biasLayer2 = np.random.uniform(size=(1, noNeuronsLayer2))


    noEpochs = 500
    learningRate = 0.3


    prediction_errors = []
    epo = []


    # Train the network for noEpochs
    for i in range(noEpochs):

        # Forward Propagation
        hidden_layer_output = forwardPropagationLayer(points, weightsLayer1, biasLayer1)
        predicted_output = forwardPropagationLayer(hidden_layer_output, weightsLayer2, biasLayer2)

        # Backpropagation
        bkProp_error = labels - predicted_output
        d_predicted_output = bkProp_error * tanhDerivative(predicted_output)

        errors = bkProp_error**2

        errors_sum = 0

        j = 0
        for e in errors:
            errors_sum += errors[j]
            j = j + 1
        errors_sum = errors_sum / 2 * len(errors)


        error_hidden_layer = d_predicted_output.dot(weightsLayer2.T)
        d_hidden_layer = error_hidden_layer * tanhDerivative(hidden_layer_output)

        # Updating Weights and Biases
        weightsLayer2 = weightsLayer2 + hidden_layer_output.T.dot(d_predicted_output) * learningRate
        biasLayer2 = biasLayer2 + np.sum(d_predicted_output, axis=0, keepdims=True) * learningRate

        weightsLayer1 = weightsLayer1 + points.T.dot(d_hidden_layer) * learningRate
        biasLayer1 = biasLayer1 + np.sum(d_hidden_layer, axis=0, keepdims=True) * learningRate

        prediction_errors.append(errors_sum)
        epo.append(i)

        if errors_sum < 0.01:
            print("min epoch num is: " + str(i))
            print("predicted output is: " + str(predicted_output))
            print("target output is: " + str(labels))
            break

    # Print weights and bias
    #print("weightsLayer1 = {}".format(weightsLayer1))
    #print("biasesLayer1 = {}".format(biasLayer1))

    #print("weightsLayer2 = {}".format(weightsLayer2))
    #print("biasLayer2 = {}".format(biasLayer2))
    plt.plot(epo, prediction_errors)
    plt.xlabel('epoch')
    plt.ylabel('prediction error')

    plt.show()

    # Display the results
    for i in range(len(labels)):
        outL1 = forwardPropagationLayer(points[i], weightsLayer1, biasLayer1)
        outL2 = forwardPropagationLayer(outL1, weightsLayer2, biasLayer2)

        print("Input = {} - Predict = {} - Label = {}".format(points[i], outL2, labels[i]))

if __name__ == "__main__":
    main()
