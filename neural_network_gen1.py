import numpy as np
import neural_network_kernels1
import data_load_MNIST

# Load Data    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print('Loading Data')
X_train, Y_train, X_test, Y_test = data_load_MNIST.load_data()

# Train Network  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
layers = neural_network_kernels1.train_Network(X_train, Y_train)

# Test Network>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
neural_network_kernels1.test_Network(X_test, Y_test, layers)
