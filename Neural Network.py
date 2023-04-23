from math import *
import numpy as np
import matplotlib.pyplot as plt

training = np.array([[0,0,0,0],
              [0,0,0,1],
              [0,0,1,0],
              [0,0,1,1],
              [0,1,0,0],
              [0,1,0,1],
              [0,1,1,0],
              [0,1,1,1],
              [1,0,0,0],
              [1,0,0,1],
              [1,0,1,0]])
trainingLabels = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

testing = np.array([[0,0,1,0],
              [0,1,0,0],
              [0,1,1,1],
              [1,0,0,1],
              [1,0,1,0]])
testingLabels = np.array([[2],[4],[7],[9],[10]])


highestLabel = max(trainingLabels)
lowestLabel = min(trainingLabels)

Range = highestLabel - lowestLabel

trainingLabels = np.array([(i - lowestLabel) / Range for i in trainingLabels])
testingLabels = np.array([(i - lowestLabel) / Range for i in testingLabels])


def Sigmoid(num):
    return 1 / (1 + np.exp(-num))

def SigmoidDerivative(num):
    return num * (1 - num)

def MeanSquaredError(labels, output):
    out = np.mean((labels - output)**2)
    return out


hiddenLayers = [4, 3]

hiddenActivation, hiddenActivationPrime = Sigmoid, SigmoidDerivative
outputActivation, ouputActivationPrime = Sigmoid, SigmoidDerivative


weights = []

weights.append(np.random.rand(training.shape[1], hiddenLayers[0]))

if len(hiddenLayers) > 1:
    for i in range(len(hiddenLayers) - 1):
        weights.append(np.random.rand(hiddenLayers[i], hiddenLayers[i+1]))

weights.append(np.random.rand(hiddenLayers[-1], 1))


def ForwardPropagation(training, weights):
    forward = []

    forward.append(hiddenActivation(np.dot(training, weights[0])))
    
    if len(hiddenLayers) > 1:
        for i in range(len(hiddenLayers) - 1):
            forward.append(hiddenActivation(np.dot(forward[i], weights[i + 1])))

    output = outputActivation(np.dot(forward[-1], weights[-1]))

    return forward, output


def BackPropagation(output, inputs, labels, weights, forward):
    d_Weights = []

    error = 2*(labels - output) * hiddenActivationPrime(output)
    d_Weights.append(np.dot(forward[-1].T, (error)))

    for i in range(len(hiddenLayers) - 1, 0, -1):
        error = np.dot(error, weights[i + 1].T) * hiddenActivationPrime(forward[i])
        d_Weights.insert(0, np.dot(forward[i - 1].T, error))

    error = np.dot(error, weights[1].T) * hiddenActivationPrime(forward[0])
    d_Weights.insert(0, np.dot(inputs.T, error))

    for i in range(len(weights)):
        weights[i] += d_Weights[i]


    return weights


def Train(data, labels, weights, epochs):
    errors = []

    for i in range(epochs):
        forward, output = ForwardPropagation(data, weights)
        errors.append(MeanSquaredError(labels, output))
        weights = BackPropagation(output, data, labels, weights, forward)
    plt.plot(errors)

    return weights


def Predict(data, labels, weights):
    forward, output = ForwardPropagation(data, weights)

    MeanAbsoluteError = np.mean(abs(output - labels))

    return "The Mean Absolute Error Is: " + str(MeanAbsoluteError)

weights = Train(training, trainingLabels, weights, 100000)

print(Predict(testing, testingLabels, weights))
plt.show()