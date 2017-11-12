"""
Simple implementation of a neural network
"""
import numpy as np
from scipy import optimize


def sigmoid(value):
    return 1. / (1 + np.exp(-value))

def sigmoid_prime(value):
    return np.exp(-value) / ((1 + np.exp(-value)) ** 2)

class NeuralNetwork(object):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, \
                eval_function=sigmoid, eval_prime_function=sigmoid_prime,
                l=0.0001):
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
        self.l = l #lambda for cost evaluation
        #Initialize Weights
        self.weights_1 = np.random.randn(self.input_layer_size, \
                                        self.hidden_layer_size)
        self.weights_2 = np.random.randn(self.hidden_layer_size, \
                                        self.output_layer_size)
        self.eval_function = eval_function
        self.eval_prime_function = eval_prime_function


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

    def cost_function_prime(self, X, y):
        """Compute cost derivative with respect to the weights

        Args:
            X (np.matrix): Input Matrix, must be of size self.input_layer_size
            y (): output vector

        """
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.eval_prime_function(self.z3))
        dJdW2 = np.dot(self.hidden_layer.T, delta3)

        delta2 = np.dot(delta3, self.weights_2.T)*self.eval_prime_function(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def cost_function_prime_with_penalty(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        dJdW1 = dJdW1/X.shape[0] + self.l * self.weights_1
        dJdW2 = dJdW2/X.shape[0] + self.l * self.weights_2
        return dJdW1, dJdW2


    def cost(self, X, y):
        """Compute cost for given input considering existing weights

        Args:
            X (np.matrix): Input Matrix, must be of size self.input_layer_size
            y (): output vector

        """
        yHat = self.forward(X)
        return 0.5 * sum((y - yHat) ** 2)

    def cost_with_pentalty(self, X, y):
        """
        Cost function that introduces penalties for high weight vectors
        """
        return self.cost(X,y)/X.shape[0] + (self.l/2) * (np.sum(self.weights_1**2) + np.sum(self.weights_2**2))


    def get_weights(self):
        """
        Concatenate weights_1 and weights_2 into a vector
        """
        return np.concatenate((self.weights_1.ravel(), self.weights_2.ravel()))

    def set_weights(self, weights):
        """
        Update weight values.
        Args:
            weights: contains first the W1, then the W2 vector
        """
        weights_1_start = 0
        weights_1_end = self.input_layer_size * self.hidden_layer_size
        self.weights_1 = np.reshape(weights[weights_1_start : weights_1_end],
                                    (self.input_layer_size, self.hidden_layer_size))

        weights_2_end = weights_1_end + self.hidden_layer_size * self.output_layer_size
        self.weights_2 = np.reshape(weights[weights_1_end : weights_2_end],
                                    (self.hidden_layer_size, self.output_layer_size))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime_with_penalty(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def train(self, trainX, trainY, testX, testY):
        """
        Train the NN with BFGS function
        """
        def cost_function_wrapper(weights, X, y):
            "Wrapper of our cost function to be used in scipy.optimize.minimize method"
            self.set_weights(weights)
            cost = self.cost_with_pentalty(trainX, trainY)
            gradient = self.compute_gradients(trainX, trainY)

            return cost, gradient

        def callbackfunc(weights):
            self.set_weights(weights)
            self.J.append(self.cost_with_pentalty(trainX, trainY))
            self.testJ.append(self.cost_with_pentalty(testX, testY))

        self.J = []
        self.testJ = []

        initial_weights = self.get_weights()
        options = {'maxiter':150, 'disp':True,}
        _res = optimize.minimize(cost_function_wrapper,
                                initial_weights,
                                jac=True,
                                method='BFGS',
                                args=(trainX, trainY),
                                options=options,
                                callback = callbackfunc,
                                )
        self.set_weights(_res.x)
        self.optimization_result = _res
