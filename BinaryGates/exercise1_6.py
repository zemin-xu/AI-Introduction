import numpy as np
import matplotlib.pyplot as plt

def sigmoid(n):

    # Define the sigmoid function as activation function
    return 1.0/(1.0 + np.exp(-n))

def sigmoidDerivative(n):

    return n*(1-n)

def forwardPropagationLayer(p, weights, biases):

    a = None  # the layer output

    # Multiply weights with the input vector (p) and add the bias   =>  n
    n = np.dot(p, weights) + biases

    # Pass the result to the activation function  =>  a
    a = sigmoid(n)

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


    noEpochs = 20000
    learningRate = 0.3

    times = []
    min_steps = []

    step_num = 101

    for t in range(step_num):
        times.append(t)

        weightsLayer1 = np.random.uniform(size=(inputSize, noNeuronsLayer1))
        weightsLayer2 = np.random.uniform(size=(noNeuronsLayer1, noNeuronsLayer2))

        biasLayer1 = np.random.uniform(size=(1, noNeuronsLayer1))
        biasLayer2 = np.random.uniform(size=(1, noNeuronsLayer2))

        # Train the network for noEpochs
        for i in range(noEpochs):

            # Forward Propagation
            hidden_layer_output = forwardPropagationLayer(points, weightsLayer1, biasLayer1)
            predicted_output = forwardPropagationLayer(hidden_layer_output, weightsLayer2, biasLayer2)

            # Backpropagation
            bkProp_error = labels - predicted_output

            # update sum of errors
            errors = bkProp_error**2

            errors_sum = 0

            j = 0
            for e in errors:
                errors_sum += errors[j]
                j = j + 1
            errors_sum = errors_sum / 2 * len(errors)

            d_predicted_output = bkProp_error * sigmoidDerivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(weightsLayer2.T)
            d_hidden_layer = error_hidden_layer * sigmoidDerivative(hidden_layer_output)

            # Updating Weights and Biases
            weightsLayer2 = weightsLayer2 + hidden_layer_output.T.dot(d_predicted_output) * learningRate
            biasLayer2 = biasLayer2 + np.sum(d_predicted_output, axis=0, keepdims=True) * learningRate

            weightsLayer1 = weightsLayer1 + points.T.dot(d_hidden_layer) * learningRate
            biasLayer1 = biasLayer1 + np.sum(d_hidden_layer, axis=0, keepdims=True) * learningRate

            if errors_sum < 0.01:
                print("min epoch num is: " + str(i))
                break

        min_steps.append(i)


    mean = np.median(min_steps)
    print("mean minimum step:" + str(mean))

    for y in range(step_num):
        min_steps[y] = (min_steps[y] - mean) / mean


    plt.ylim(top=0.6, bottom=-0.6)
    plt.plot(times, min_steps, 'ro')
    plt.xlabel('epoch')
    plt.ylabel('minimum steps')
    plt.show()

    # Display the results
    for i in range(len(labels)):
        outL1 = forwardPropagationLayer(points[i], weightsLayer1, biasLayer1)
        outL2 = forwardPropagationLayer(outL1, weightsLayer2, biasLayer2)

        # print("Input = {} - Predict = {} - Label = {}".format(points[i], outL2, labels[i]))

if __name__ == "__main__":
    main()
