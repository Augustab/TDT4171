import unittest
import numpy as np
import pickle
import os

__author__ = "August Asheim Birkeland"

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        self.hidden_layer = hidden_layer
        self.hidden_units = 25
        # I split between the initialization of a model with hidden layer or not.
        if hidden_layer:
            self.nodes = [[], [], 0]
            # I initialize the weights randomly between -0.5 and 0.5 with np.random.uniform ( + 1 because of bias)
            self.weights = [np.random.uniform(-0.5, 0.5, (self.hidden_units, input_dim + 1)), np.random.uniform(-0.5, 0.5, self.hidden_units + 1)]
        else:
            self.nodes = [[], 0]
            self.weights = [np.random.uniform(-0.5, 0.5, input_dim + 1)]
        # This is the value of α on Line 25 in Figure 18.24. I have lowered the learning-rate
        self.lr = 1e-2 # Jeg erfarte at dette var        learning-raten som ga best resultat!!
        # Og det står jo på piazza at man kan justere denne.
        self.epochs = 400 # Epochs har jeg ikke økt siden dette øker kjøretiden
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None


    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'oving5\data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        print(self.weights[0])
        for e in range(self.epochs):
            #print(e/400) # This is just to keep track of the progress
            for x, y in zip(self.x_train, self.y_train):
                # I set the input-nodes to the given x-input
                self.nodes[0] = x
                # I append a -1 to represent the bias.
                self.nodes[0] = np.append(self.nodes[0], -1)
                # I store the calculated INJ-value to be used later.
                inj_middle_perceptron = np.dot(self.weights[0], self.nodes[0])
                # The hidden layer has to take into account the bias element
                inj_middle = np.append(inj_middle_perceptron, -1)
                # This is the line that calculates the activation of layer 1.
                self.nodes[1] = self.sigmoid(np.dot(self.weights[0], self.nodes[0]))
                # Notice that i use numpy.dot so that the runtime gets faster!!
                if self.hidden_layer:
                    # Append a bias element to the hidden layer
                    self.nodes[1] = np.append(self.nodes[1], -1)
                    # I store the calculated INJ-value to be used later
                    inj_last = np.dot(self.weights[1], self.nodes[1])
                    # I calculate the activation of the output-node
                    self.nodes[2] = self.sigmoid(inj_last)
                    # FROM HERE WE DO BACKPROP
                    # Delta_out is the value used to update weights between hidden and output-layer
                    delta_out = self.sigmoid_derivative(inj_last) * (y - self.nodes[2])
                    # Delta_hidden is the value used to update weights between input and hidden-layer
                    delta_hidden = self.sigmoid_derivative(inj_middle)*self.weights[1]*delta_out
                    # I update the weights between hidden and output-layer based on delta-out, lr and their activation.
                    self.weights[1] = np.add(self.weights[1], self.lr*delta_out*self.nodes[1])
                    for d in range(len(delta_hidden)-1):
                        # I update the weights between input and hidden-layer based on delta-hidden, lr and their activation.
                        self.weights[0][d] = np.add(self.weights[0][d], self.lr * delta_hidden[d] * self.nodes[0])
                else:
                    # I calculate the delta_out value of the perceptron
                    delta_out = self.sigmoid_derivative(inj_middle_perceptron) * (y - self.nodes[1])
                    # I update the single set of weights of the perceptron
                    self.weights[0] = np.add(self.weights[0], self.lr * delta_out * self.nodes[0])

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # This is basically the same as the start of the train-function, so i dont have to comment
        self.nodes[0] = x
        self.nodes[0] = np.append(self.nodes[0], -1)
        self.nodes[1] = self.sigmoid(np.dot(self.weights[0], self.nodes[0]))

        if self.hidden_layer:
            self.nodes[1] = np.append(self.nodes[1], -1)
            inj_last = np.dot(self.weights[1], self.nodes[1])
            self.nodes[2] = self.sigmoid(inj_last)
            return self.nodes[2]
        return self.nodes[1]

    def sigmoid(self, x):
        """Activation function"""
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of activation function"""
        return self.sigmoid(x)*(1-self.sigmoid(x))


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print("Accuracy", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("accuracy", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    TEST = TestAssignment5()
    TEST.setUp()
    TEST.test_one_hidden()
    TEST.test_perceptron()


