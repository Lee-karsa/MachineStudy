"""
@Project: tensorflow_neural_network
@Description: TODO a simple neural network and study the back propagation network
@Time:2021/6/3 10:55
@Author: LeeRoc
"""

import numpy as np

class Simple_NeuralNetwork:
    def __init__(self):
        """
        To intialise a single weight,
        the dimension of weights that could intialise by inputs' dimention need to define
        @rtype: null
        """
        self.Weight = np.random.random((3, 1))
        self.Cost = []

    def __Adder(self, Input):
        """
        @param: Input is a np.array, and here require a [4,3] as a input array,
         the dimension is [4,3] and to multiply [3,1] weights
        @rtype: np.array, dimension is [4,1]
        """
        return np.dot(Input, self.Weight)

    def __Sigmod(self, Input):
        """
        To calculate the sigmod
        @rtype: np.array, this function doesn't change the dimension of parameter of Input
        """
        return 1 / (1 + np.exp(-Input))

    def __sigmoid_derivative(self, Input):
        """
        To calculate sigmod, get the Partial Derivative of weight
        @rtype: ny.array,[4,1]
        """
        return Input * (1 - Input)

    def Activated(self, Inputs):
        """
        this is activated function
        @rtype: np.array, this function doesn't change the dimension of parameter of Input
        """
        return self.__Sigmod(self.__Adder(Inputs))

    def Train(self, epochs, learning_rate, train_input, train_output):
        """
        @param: epochs aim at iteration to training
        @param: learning_rate between 0-1, always choose 0.1 or smaller, nickname is step of neural network
        @rtype: null
        """
        for epoch in range(epochs):
            # every iteration 's output
            # its dimension is [4,1]
            Each_output = self.Activated(train_input)

            # the error output and standard output, [4,1]-[4,1] = [4,1]
            error = np.subtract(Each_output, train_output)

            # adjust the weight, [4,3].T*[4,1]
            Adjustment = np.dot(train_input.T, error * self.__sigmoid_derivative(Each_output))

            # back propagation, the result is [3,1]
            self.Weight = np.subtract(self.Weight, (np.dot(learning_rate, Adjustment)))

if __name__ == "__main__":
    simple_neural = Simple_NeuralNetwork()

    # structure the data set, include input and output set,
    # although the code of github that i cite also like this, so focking paralyse me
    # so focking light train set, insight me
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    # start train
    simple_neural.Train(10000, 0.1, training_set_inputs, training_set_outputs)

    # test
    print(simple_neural.Activated(np.array([1, 1, 0])))
