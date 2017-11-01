"""
Simple implementation of a neural network
"""
import numpy as np

def sigmoid(value):
    return 1. / (1 + np.exp(-value))

class NeuralNetwork(object):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, \
                eval_function=sigmoid):
        """Initialize the NN

        Args:
            input_layer_size (int): size of the input dimention
            hidden_layer_size (int): size of the hidden layer
            output_layer_size (int): size of the output dimention
            eval_function (function): function to evaluate neurons on
        """
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        #Initialize Weights
        self.weights_1 = np.random.randn(self.input_layer_size, \
                                        self.hidden_layer_size)
        self.weights_2 = np.random.randn(self.hidden_layer_size, \
                                        self.output_layer_size)
        self.eval_function = eval_function


    def forward(self, X):
        """ Propagate the values through the network
        z2 = X * W1
        hl = f(z2)
        z3 = hl * W2
        yHat = f(z3)

        Args:
            X (np.matrix): Input Matrix, must be of size self.input_layer_size
        """
        self.z2 = np.dot(X,  self.weights_1)
        self.hidden_layer = self.eval_function(self.z2)
        self.z3 = np.dot(self.hidden_layer, self.weights_2)
        self.yHat = self.eval_function(self.z3)
        return self.yHat
