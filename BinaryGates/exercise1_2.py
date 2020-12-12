import sys

def activationFunction(n):
    # hardlim
    if n >= 0:
        n = 1
    else:
        n = 0
    return n

def forwardPropagation(p, weights, bias):
    a = None  # the neuron output
    n = p[0] * weights[0] + p[1] * weights[1] + bias
    a = activationFunction(n)
    return a

def main():
    # Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    # The network should receive as input two values (0 or 1) and should predict the target output

    # Input data
    P = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # Labels to perform OR gate
    t = [0, 1, 1, 1]

    weights = [0, 0]
    bias = 0
    epochs = 10

    for ep in range(epochs):
        correct = 0
        for i in range(len(t)):
            pred = forwardPropagation(P[i], weights, bias)

            error = t[i] - pred

            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]

            bias = bias + error

            # count error times
            if error == 0:
                correct += 1

        # if all errors are 0, then this is the minimal epochs
        if correct == len(t):
            print("min epochs = " + str(ep))
            print("weights[0] = " + str(weights[0]))
            print("weights[1] = " + str(weights[1]))
            print("bias = " + str(bias))
            break

    return


if __name__ == "__main__":
    main()
