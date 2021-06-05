# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os


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

        if hidden_layer:
            self.nodes = [[], [], 0]
            first = np.random.uniform(-0.5, 0.5, (self.hidden_units, input_dim + 1))
            second = np.random.uniform(-0.5, 0.5, self.hidden_units + 1)
            self.weights = [first, second]
        else:
            self.nodes = [[0 for x in range(input_dim + 1)], 0]
            self.weights = [np.random.uniform(-0.5, 0.5, input_dim + 1)]

        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3
        self.epochs = 400
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
            for x, y in zip(self.x_train, self.y_train):
                self.nodes[0] = x
                self.nodes[0] = np.append(self.nodes[0], -1)
                self.nodes[1] = self.sigmoid(np.dot(self.weights[0], self.nodes[0]))
                if self.hidden_layer:
                    self.nodes[1] = np.append(self.nodes[1], -1)
                    inj_last = np.dot(self.weights[1], self.nodes[1])
                    self.nodes[2] = self.sigmoid(inj_last)
                    # Now i have propagated the inputs forward.
                    # Time to propagate deltas backward.
                    # EVT self.sigmoid_derivative(inj_last) * (y - self.nodes[2])
                    delta_out = self.sigmoid_derivative(self.nodes[2]) * (y - self.nodes[2])
                    delta_hidden = self.sigmoid_derivative(self.nodes[1])*self.weights[1]*delta_out
                    self.weights[1] = np.add(self.weights[1], self.lr*delta_out*self.nodes[1])
                    '''for count, delta in enumerate(delta_hidden):
                        for i in range(self.weights[0][0].size):
                            #print("self.weights[0][i][count]",self.weights[0][i][count])
                            #print("delta_out[count]", delta_out[count])
                            #print("self.nodes[0][i]", self.nodes[0][i])
                            self.weights[0][count-1][i] = self.weights[0][count-1][i] + self.lr * delta_hidden[count] * self.nodes[0][i]'''
                    for count, delta in enumerate(delta_hidden):
                        self.weights[0][count - 1] = np.add(self.weights[0][count-1], self.lr * delta_hidden[count] * self.nodes[0])
                    #self.weights[0] = (np.add(np.transpose(self.weights[0]),self.lr*self.nodes[0]*delta_hidden))
                else:
                    delta_out = self.sigmoid_derivative(self.nodes[1]) * (y - self.nodes[1])
                    self.weights[0] = np.add(self.weights[0], self.lr * delta_out * self.nodes[0])




    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
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
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
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
        print("ACCURACYMANNEN", accuracy)
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


