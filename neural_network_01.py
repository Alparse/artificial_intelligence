import numpy as np
import matplotlib.pyplot as plt
import pickle
import data_load_MNIST

networkStructure = np.array([784, 300, 30, 10])

network_activations = {0: 'input', 1: 'tanh', 2: 'tanh', 3: 'sig'}

numLayers = np.size(networkStructure) - 1

output_layer = numLayers

input_layer = 0

alpha = 1

lambd = 3

randmag = .3

iterations = 1000

cost_target = .5


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A


def relu_derivative(Z):
    dZ = np.array(Z, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def tanhf(Z):
    return np.tanh(Z)


def tanhf_derivative(Z):
    return 1.0 - np.square(np.tanh(Z))


def compute_cost(AL):
    return -1 / m * np.sum((Y * np.log(AL)) + (1 - Y) * np.log(1 - AL))


def activation(i, Z):
    activation_function = network_activations[i]
    if activation_function == 'sig':
        return sigmoid(Z)
    if activation_function == 'relu':
        return relu(Z)
    if activation_function == 'tanh':
        return tanhf(Z)


def activation_derivative(i, Z):
    activation_function = network_activations[i]
    if activation_function == 'sig':
        return sigmoid_derivative(Z)
    if activation_function == 'relu':
        return relu_derivative(Z)
    if activation_function == 'tanh':
        return tanhf_derivative(Z)


def initialize_train_network():
    print('Initializing Training Network')

    global X
    X = X_train.T
    print('Shape of X ', np.shape(X))
    global Y
    Y = Y_train.T
    global m
    m = np.size(X_train.T, 1)
    print('m ', m)

    global layers
    layers = {'A0': X}
    print("A0 ", np.shape(layers['A0']))

    assert (layers['A0'].shape == (networkStructure[0], m))

    for i in range(1, numLayers + 1):
        layers['W' + str(i)] = np.random.randn(networkStructure[i], networkStructure[i - 1]) * randmag
        layers['b' + str(i)] = np.zeros((networkStructure[i], 1))
        layers['Z' + str(i)] = np.zeros((networkStructure[i], m))
        layers['A' + str(i)] = np.zeros((networkStructure[i], m))

        assert (layers['W' + str(i)].shape == (networkStructure[i], networkStructure[i - 1]))
        assert (layers['b' + str(i)].shape == (networkStructure[i], 1))
        assert (layers['Z' + str(i)].shape == (networkStructure[i], m))
        assert (layers['A' + str(i)].shape == (networkStructure[i], m))


def initialize_test_network():
    print('Initializing Test Network')
    global X
    X = X_test.T
    print(np.shape(X))
    global Y
    Y = Y_test.T
    global m
    m = np.size(X_test.T, 1)
    print(m)
    global layers
    layers['A0'] = X
    print("A0 ", np.shape(layers['A0']))
    assert (layers['A0'].shape == (networkStructure[0], m))

    for i in range(1, numLayers + 1):
        layers['Z' + str(i)] = np.zeros((networkStructure[i], m))
        layers['A' + str(i)] = np.zeros((networkStructure[i], m))

        assert (layers['Z' + str(i)].shape == (networkStructure[i], m))
        assert (layers['A' + str(i)].shape == (networkStructure[i], m))


def forward_propagate():
    print('Forward Propogating')
    global layers
    for i in range(1, numLayers + 1):
        layers['Z' + str(i)] = np.dot(layers['W' + str(i)], layers['A' + str(i - 1)]) + layers['b' + str(i)]
        layers['A' + str(i)] = activation(i, layers['Z' + str(i)])


def back_propagate():
    print('Back Propogating')
    global gradients

    gradients = {'dZ' + str(output_layer): layers['A' + str(output_layer)] - Y}
    gradients['dW' + str(output_layer)] = 1 / m * np.dot(gradients['dZ' + str(output_layer)],
                                                         layers['A' + str(output_layer - 1)].T)
    gradients['db' + str(output_layer)] = 1 / m * np.sum(gradients['dZ' + str(output_layer)], axis=1, keepdims=True)

    for i in range(output_layer - 1, input_layer, -1):
        gradients['dZ' + str(i)] = np.dot(layers['W' + str(i + 1)].T,
                                          gradients['dZ' + str(i + 1)]) * activation_derivative(i, layers['Z' + str(i)])
        gradients['dW' + str(i)] = 1 / m * np.dot(gradients['dZ' + str(i)], layers['A' + str(i - 1)].T) + (
                lambd * layers['W' + str(i)]) / m
        gradients['db' + str(i)] = 1 / m * np.sum(gradients['dZ' + str(i)], axis=1, keepdims=True)

    for i in range(output_layer, input_layer, -1):
        layers['W' + str(i)] = layers['W' + str(i)] - alpha * gradients['dW' + str(i)]
        layers['b' + str(i)] = layers['b' + str(i)] - alpha * gradients['db' + str(i)]


# Load Data    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print('Loading Data')
X_train, Y_train, X_test, Y_test = data_load_MNIST.load_data()

# Train Network  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
initialize_train_network()

cost_curve = []

for i in range(1, iterations):
    print(i)
    forward_propagate()
    back_propagate()
    cost = -1 / m * np.sum(
        (Y * np.log(layers['A' + str(output_layer)])) + (1 - Y) * np.log(1 - layers['A' + str(output_layer)]))
    print(cost)
    cost_curve.append(cost)
    if cost <= cost_target:
        break

# Test Network>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
initialize_test_network()
forward_propagate()
cost = -1 / m * np.sum(
    (Y * np.log(layers['A' + str(output_layer)])) + (1 - Y) * np.log(1 - layers['A' + str(output_layer)]))
print(np.shape(layers['A' + str(output_layer)]))
print(np.shape(layers['A' + str(output_layer)]))
print(np.shape(Y))

Yh = layers['A' + str(output_layer)]

print("Yh ", np.shape(Yh))

for i in range(m):
    for n in range(10):
        if Yh[n, i] > .5:
            Yh[n, i] = 1
        else:
            Yh[n, i] = 0

errors = 0
correct = 0

for i in range(m):
    print("example ", i, "Y ", Y[:, i])
    print("example ", i, "Yh", np.around((layers['A' + str(output_layer)][:, i]), decimals=0))
    for n in range(10):
        if Y[n, i] == 1 and Y[n, i] != Yh[n, i]:
            errors = errors + 1
            break
    correct = correct + 1

print("cost ", cost)
print("count ", correct)
print("errors ", errors)
print("Accuracy", (m - errors) / m * 100, "%")

layers_save = open('layers_file.p', 'wb')
pickle.dump(layers, layers_save, -1)
layers_save.close()

plt.plot(cost_curve)
plt.ylabel('Cost')
plt.show()
