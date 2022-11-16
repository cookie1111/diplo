import numpy as np
from itertools import combinations as comb
import concurrent.futures as ft
from typing import Callable
from math import floor


def least_squares_error(z: np.ndarray, y: np.ndarray) -> float:
    """
    Least squares error calculated:
    err = sum((y-z)^2)

    :param z: calculated predictions
    :param y: ground truth
    :return: least squares error
    """
    if (z is None) or (y is None):
        print("One of the inputs is None")
        return -1

    if len(z) != len(y):
        print(f"{len(z)} is not the same as {len(y)}")
        return -1

    return np.sum(np.square(y-z))


class GMDHLayer:

    def __init__(self, inputs: np.ndarray | "GMDHLayer", threshold: float = None, first_layer: bool = False,
                 parallel: bool = False, workers: int = 1, max_neurons: int = 128) -> None:
        """
        initializes the layer

        :param inputs: previous layer nodes or original input, where previous layer is either GMDHLayer object or
        collection of input variables in a np.array
        :param threshold: a threshold to get rid of neurons that are not good enough, so we don't waste our time with
        comparing and space for storing, if not set its set to median of previous layer
        :param first_layer: check if its first layer since that would mean we don't get a previous layer but numpy array
        :param parallel: do we want to parallelize the workload
        :param workers: number of workers to use in parallelization
        :param max_neurons: max amount of neurons in layer
        """
        self.first_layer = first_layer
        self.max_neurons = max_neurons
        self.neurons = [PolyLeastSquares(inputs, first=first_layer) for inputs in (comb(range(len(inputs)), 2) if
                                                                                   not first_layer else comb(
            range(inputs.shape[1]), 2))]
        self.parallel = parallel
        self.workers = workers
        self.threshold = threshold
        self.prev_layer = inputs

    def __getitem__(self, i):
        return self.neurons[i]

    def train_layer(self, prev: np.ndarray | "GMDHLayer", y: np.ndarray, fitness_fn: Callable = least_squares_error,
                    split: float = 0.5) -> int:
        """
        Trains layer and selects, which neurons to keep from previous layers, which to replace and which new neurons
        to add
        :param prev: previous layer or input if the current layer is the first layer
        :param y: array of ground truths
        :param fitness_fn: fitness function to be used for error calculation
        :param split: ratio between training and selection sets
        """
        # entries are tuples of (fitness and neurons)
        accepted_comp = []
        if self.parallel:
            with ft.ProcessPoolExecutor(max_workers=self.workers) as executor:
                futs = [executor.submit(neuron.regression_of_function(prev, y)) for neuron in self.neurons]
                ft.wait(futs)
        else:
            # creation and addition of previous layers neurons as pass-through neurons
            if self.first_layer:
                # create through neurons of previous layer
                for neuron in range(self.prev_layer.shape[1]):
                    neuro = PolyLeastSquares(neuron, through=True, first=True)
                    err = neuro.regress_and_test(prev, y, fitness_fn, split)
                    accepted_comp.append((err[1], neuron))
            else:
                for neuron in range(len(self.prev_layer)):
                    neuro = PolyLeastSquares(neuron, through=True, first=True)
                    err = neuro.regress_and_test(prev, y, fitness_fn, split)
                    accepted_comp.append((err[1], neuron))

            # test new neurons and add them to queue if condition is satisfied
            while len(self.neurons) > 0:
                neuron = self.neurons.pop()
                error = neuron.regression_of_function(prev, y)
                if error[1] < self.threshold:
                    accepted_comp.append((error[1], neuron))

            self.neurons = accepted_comp.sort()[:self.max_neurons]

            return 1


# TODO: at a later date add the option of storing or not storing previous layers outputs
class PolyLeastSquares:

    def __init__(self, input_indexes: list[int] | int, coefficients: list[float] = None, first: bool = False,
                 through: bool = False) -> None:
        """
        Creates a polynomial least squares neuron

        :param input_indexes: index of previous layers neurons that are combined in this neuron, in case of through being
        True there is only one index here
        :param coefficients: if we want to predefine coefficients in the current neuron
        :param first: True if this is the first layer else False
        :param through: True if this is just a normal through gate
        """
        self.through = through
        if through:
            self.x1 = input_indexes
            self.x2 = None
        else:
            self.c_non_matrix = coefficients
            self.coefficients = None if not coefficients else self.to_lin_alg(coefficients)
            self.x1, self.x2 = input_indexes
            self.first = first

    @staticmethod
    def to_lin_alg(c):
        """
        converts coefficient into matrix form

        :param c:
        :return:
        """
        return c[0], np.array([c[1], c[2]]), np.array([[c[3], c[5] / 2], [c[5] / 2, c[4]]])

    def calc_quadratic(self, x1: np.ndarray, x2: np.ndarray | None = None) -> list[float]:
        """
        calculate this neurons output based on previous layers input

        :param x1: results of first input from previous layer
        :param x2: results of second input from previous layer can be None if it's a pass through neuron
        :return: result of the quadratic function
        """
        if self.through:
            return x1
        if self.c_non_matrix is None:
            return -1

        return [self.c_non_matrix[0] + self.c_non_matrix[1] * x1 + self.c_non_matrix[2] * x2 + self.c_non_matrix[3] * (
                    x1 ** 2) + self.c_non_matrix[4] * (x2 ** 2) + self.c_non_matrix[5] * x1 * x2 for x1, x2 in zip(x1,
                                                                                                                   x2)]

    def calc_quadratic_matrix(self, x1: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
        """
        calculate this neurons output based on previous layers input

        :param x1: results of first input from previous layer
        :param x2: results of second input from previous layer can be None if it's a pass through neuron
        :return: result of the quadratic function
        """
        if self.through:
            return x1
        if self.coefficients is None:
            return -1

        c = np.array([x1, x2])
        # Z = N.diag(X.dot(Y)) -> Z = (X * Y.T).sum(-1)
        uno = np.matmul(self.coefficients[2], c)
        dos = (uno.T * c.T).sum(-1)
        tres = np.dot(c.T, self.coefficients[1])
        print(tres.shape)
        return self.coefficients[0] + tres + dos

    def get_prev(self, prev: np.ndarray | GMDHLayer, matrix: bool = False) -> tuple[list[float], list[float]]:
        """
        Fetches previous layers neurons(x1 and x2) outputs

        :param prev: previous layer
        :param matrix: if True uses matrices for calculating previous layers output
        :return: results of the neurons that are to be combined
        """
        if self.first:
            if self.through:
                return prev[:, self.x1]
            x1 = prev[:, self.x1]
            x2 = prev[:, self.x2]
        else:
            if matrix:
                if self.through:
                    return prev[self.x1].calc_quadratic_matrix()
                x1 = prev[self.x1].calc_quadratic_matrix()
                x2 = prev[self.x2].calc_quadratic_matrix()
            else:
                if self.through:
                    return prev[self.x1].calc_quadratic()
                x1 = prev[self.x1].calc_quadratic()
                x2 = prev[self.x2].calc_quadratic()
        return x1, x2

    # storage of previous input vs calculating it every time...
    def regression_of_function(self, input_x1: np.ndarray , input_x2: np.ndarray | None, y: np.ndarray,
                               fitness_fn: Callable = least_squares_error) -> float:
        """
        Runs least squares regression to solve for A,B,C,D,E,F in:
         A + B * u + C * v + D * u^2 + E * v^2 + F * u * v

        :param input_x1: first inputs results
        :param input_x2: second inputs results can be None if its a pass through neuron
        :param fitness_fn:
        :param y: target variable
        :return: returns error of the function
        """


        # c = np.array([prev[self.x1],prev[self.x2]])
        A = np.array([input_x1 * 0 + 1, input_x1, input_x2, input_x1 ** 2, input_x2 ** 2, input_x1 * input_x2]).T

        # euclidean 2-norm based residual
        coeff, r, rank, s = np.linalg.lstsq(A, y)
        self.c_non_matrix = coeff
        self.coefficients = self.to_lin_alg(coeff)

        # calculating fitness

        return self.get_error(input_x1, input_x2, y, fitness_fn)

    def get_error(self, input_x1: np.ndarray, input_x2: np.ndarray | None, y: np.ndarray,
                  fitness_fn: Callable = least_squares_error) -> float:
        """
        Calculates error between the prediction and ground truth based on passed function
        :param input_x1: previous layers first neurons output
        :param input_x2:  previous layers second neurons output or None if it's a pass through layer
        :param y: ground truth for predicted variable
        :param fitness_fn: function to use for calculating the error has to have 1st argument be the predicted values
        and second argument the ground truth
        :return: calculated error between prediction and ground truth
        """
        res = self.calc_quadratic_matrix(input_x1, input_x2)
        return fitness_fn(res, y)

    def regress_and_test(self, prev: GMDHLayer | np.ndarray, y: np.ndarray, fitness_fn: Callable = least_squares_error,
                         split: float = 0.5) -> tuple[float, float]:
        """
        Regresses the neuron and calculates both train and selection error for said neuron

        :param prev: complete previous layer in case this is first layer the previous layer is inputs
        :param y: ground truth values to be predicted
        :param fitness_fn: function for calculating the error between predicted and ground truth values
        :param split: ratio between train and selection set
        :return: error on train set, error on selection set(selection set is the one that matters)
        """
        if not self.through:
            input_x1, input_x2 = self.get_prev(prev)
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            train_x2 = input_x2[:floor(len(input_x2) * split)]
            train_y = y[:floor(len(y) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = input_x2[floor(len(input_x2) * split):]
            test_y = y[floor(len(y) * split):]
            train_res = self.regression_of_function(train_x1, train_x2, train_y, fitness_fn= fitness_fn)
        else:
            input_x1 = self.get_prev(prev)
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = None
            train_y = y[:floor(len(y) * split)]
            test_y = y[floor(len(y) * split):]
            train_res = self.get_error(train_x1,None, y, fitness_fn)

        selection_res = self.get_error(test_x1, test_x2, test_y, fitness_fn)
        return train_res, selection_res
