import matplotlib.pyplot as plt


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def activationFunction(n):

    #TODO - Application 1 - Step 4b - Define the binary step function as activation function
    if n >= 0:
        n = 1
    else:
        n = 0

    return n

#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def forwardPropagation(p, weights, bias):

    a = None # the neuron output

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    n = p[0] * weights[0] + p[1] * weights[1] + bias

    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(n)

    return a
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def main():

    #Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    #The network should receive as input two values (0 or 1) and should predict the target output

    #Input data
    P = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    #Labels
    t = [0, 0, 0, 1]

    #TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    weights = [0, 0]

    #TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    bias = 0

    #TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 10

    w0 = []
    w1 = []
    e = []
    b = []
    #TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(t)):

            #TODO - Application 1 - Step 4 - Call the forwardPropagation method
            pred = forwardPropagation(P[i], weights, bias)


            #TODO - Application 5 - Compute the prediction error (error)
            error = t[i] - pred

            #TODO - Application 6 - Update the weights
            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]

            #TODO - Update the bias
            bias = bias + error

        w0.append(weights[0])
        w1.append(weights[1])
        b.append(bias)
        e.append(ep)

    # print weights and bias histogram
    line1, = plt.plot(e, w0)
    line2, = plt.plot(e, w1)
    line3, = plt.plot(e, b)
    plt.legend([line1, line2, line3], ['Weight 1', 'Weight 2', 'Bias'])
    plt.title('Change of weights and bias')
    plt.xlabel('epoch')
    plt.ylabel('value')

    plt.show()

    #TODO - Application 1 - Print weights and bias
    print("weights[0] = " + str(weights[0]))
    print("weights[1] = " + str(weights[1]))
    print("bias = " + str(bias))

    # TODO - Application 1 - Step 7 - Display the results
    plt.show()

    return
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
if __name__ == "__main__":
    main()
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################