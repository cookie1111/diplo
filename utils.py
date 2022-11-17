import numpy as np
from itertools import combinations as comb
import concurrent.futures as ft
from typing import Callable, Union
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

    return np.sum(np.square(y - z))


def mean_square_error(z: np.ndarray, y: np.ndarray) -> float:
    """
    Mean square error calculated:
    err = sum((y-z)^2)/len(y)

    :param z: calculated predictions
    :param y: ground truth
    :return: mean square error
    """
    if (z is None) or (y is None):
        print("One of the inputs is None")
        raise "yeah nah buddy0"
        return -1

    if len(z) != len(y):
        print(f"{len(z)} is not the same as {len(y)}")
        raise "yeah nah buddy0"
        return -1

    return np.sum(np.square(y - z)) / len(y)


class GMDH:
    def __init__(self, inputs, y, max_neurons_per_layer=128, err_fn=least_squares_error, err_leeway=1,
                 split_train_select=0.75):
        self.layers = []
        self.inputs = inputs
        self.y = y
        self.max_neurons = max_neurons_per_layer
        self.err_fn = err_fn
        self.err_leeway = err_leeway
        self.split_train_select = split_train_select
        self.can_evaluate = False

    def train(self):
        loss_cnt = 0
        print(f"Starting training")
        self.layers.append(GMDHLayer(self.inputs, 200000, first_layer=True))
        min_loss = self.layers[-1].train_layer(prev=self.inputs, y=self.y, fitness_fn=self.err_fn,
                                               split=self.split_train_select)
        while True:
            print(f"Training layer {len(self.layers)}")
            k = self.layers.append(GMDHLayer(inputs=self.layers[-1], first_layer=False))
            cur_loss = self.layers[-1].train_layer(prev=self.layers[-2], y=self.y, fitness_fn=self.err_fn,
                                                   split=self.split_train_select)
            if min_loss <= cur_loss:
                if loss_cnt == self.err_leeway:
                    print(f"Layer {len(self.layers)} has triggered the train stop condition")
                    break
                else:
                    print(f"Layer {len(self.layers)} has underperformed")
                    loss_cnt = loss_cnt + 1
            else:
                min_loss = cur_loss
                loss_cnt = 0

        # neuron with the smallest error is always first in the layer
        self.layers = self.layers[:-loss_cnt]
        self.layers[-1].reduce_to_best_neuron()

        self.can_evaluate = True
        print("Model has finished training")

    def test(self, inputs):
        print(self.layers)
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs



class GMDHLayer:

    def __init__(self, inputs: Union[np.ndarray, "GMDHLayer"], threshold: float = None, first_layer: bool = False,
                 parallel: bool = False, workers: int = 1, max_neurons: int = 128) -> None:
        """
        initializes the layer

        :param inputs: previous layer nodes or original input, where previous layer is either GMDHLayer object or
        collection of input variables in a np.ndarray
        :param threshold: a threshold to get rid of neurons that are not good enough, so we don't waste our time with
        comparing and space for storing, if not set it's set to median of previous layer
        :param first_layer: check if its first layer since that would mean we don't get a previous layer but numpy array
        :param parallel: do we want to parallelize the workload
        :param workers: number of workers to use in parallelization
        :param max_neurons: max amount of neurons in layer
        """
        # self.index = index
        # print(f"GMDHLayer: {inputs}")
        self.first_layer = first_layer
        self.max_neurons = max_neurons
        self.neurons = [PolyLeastSquares(inputs, inp, first=self.first_layer) for inp in
                        (comb(range(len(inputs)), 2) if not self.first_layer else comb(range(inputs.shape[1]), 2))]
        # print("oop")
        self.parallel = parallel
        self.workers = workers
        self.threshold = threshold
        self.prev_layer = inputs

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, i):
        # print(i)
        # print(f"neurons: {self.neurons}")
        # print(f"you want dis? {self.neurons[i]}")
        return self.neurons[i][1]

    def train_layer(self, prev: Union[np.ndarray, "GMDHLayer"], y: np.ndarray,
                    fitness_fn: Callable = least_squares_error,
                    split: float = 0.5) -> int:
        """
        Trains layer and selects, which neurons to keep from previous layers, which to replace and which new neurons
        to add

        :param prev: previous layer or input if the current layer is the first layer
        :param y: array of ground truths
        :param fitness_fn: fitness function to be used for error calculation
        :param split: ratio between training and selection sets
        :return: the minimum error of all the neurons
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
                    neuro = PolyLeastSquares(self.prev_layer, neuron, through=True, first=True)
                    # print(split)
                    err = neuro.regress_and_test(prev, y, fitness_fn, split)
                    accepted_comp.append((err[1], neuro))
                accepted_comp.sort()
                # print(f"first: {neuro}")
                threshold = accepted_comp[floor(len(accepted_comp) / 2)][0]
            else:
                for neuron in range(len(self.prev_layer)):
                    neuro = PolyLeastSquares(self.prev_layer, neuron, through=True, first=False)
                    err = neuro.regress_and_test(prev, y, fitness_fn, split)
                    # print(f"current neuron:{neuro}")
                    accepted_comp.append((err[1], neuro))
                accepted_comp.sort()
                threshold = accepted_comp[floor(len(accepted_comp) / 2)][0]

            # test new neurons and add them to queue if condition is satisfied
            while len(self.neurons) > 0:
                neuron = self.neurons.pop()
                # print(f"uno neuron{neuron}")
                error = neuron.regress_and_test(prev, y, fitness_fn, split)
                # print(threshold, error)
                if error[1] < threshold:
                    # print(f"YOU ARE HIRED")
                    accepted_comp.append((error[1], neuron))
            # print(f"Accepted trained:{accepted_comp[0]},{accepted_comp[1]}")
            accepted_comp.sort()
            self.neurons = accepted_comp
            # print(self.neurons)

            return self.neurons[0][0]

    def reduce_to_best_neuron(self):
        self.neurons = [self.neurons[0]]

    def forward(self, inputs):
        print(self.neurons)
        res = []
        for n in self.neurons:
            res.append(n[1].forward_pass(inputs))
        return res



class PolyLeastSquares:

    def __init__(self, prev, input_indexes: list[int] | int, coefficients: list[float] = None, first: bool = False,
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
        self.prev = prev
        if through:
            self.x1 = input_indexes
            self.x2 = None
        else:
            self.c_non_matrix = coefficients
            self.coefficients = None if not coefficients else self.to_lin_alg(coefficients)
            self.x1, self.x2 = input_indexes
        # print(f"New wave{first}")
        self.first = first
        self.results = [None, None]

    @staticmethod
    def to_lin_alg(c):
        """
        converts coefficient into matrix form

        :param c:
        :return:
        """
        return c[0], np.array([c[1], c[2]]), np.array([[c[3], c[5] / 2], [c[5] / 2, c[4]]])

    def calc_quadratic(self) -> list[float]:
        """
        calculate this neurons output based on previous layers input

        :param x1: results of first input from previous layer
        :param x2: results of second input from previous layer can be None if it's a pass through neuron
        :return: result of the quadratic function
        """
        x1 = self.get_prev(matrix=False)
        if type(x1) is tuple:
            x1 = x1[0]
            x2 = x1[1]

        if self.through:
            return x1
        if self.c_non_matrix is None:
            return -1

        return [self.c_non_matrix[0] + self.c_non_matrix[1] * x1 + self.c_non_matrix[2] * x2 + self.c_non_matrix[3] * (
                x1 ** 2) + self.c_non_matrix[4] * (x2 ** 2) + self.c_non_matrix[5] * x1 * x2 for x1, x2 in zip(x1, x2)]

    def calc_quadratic_matrix(self) -> np.ndarray:
        """
        calculate this neurons output based on previous layers input

        :param x1: results of first input from previous layer
        :param x2: results of second input from previous layer can be None if it's a pass through neuron
        :return: result of the quadratic function
        """
        x1 = self.get_prev(matrix=True)
        if type(x1) is tuple:
            x1 = x1[0]
            x2 = x1[1]

        if self.through:
            return x1
        if self.coefficients is None:
            return -1

        c = np.array([x1, x2])
        # Z = N.diag(X.dot(Y)) -> Z = (X * Y.T).sum(-1)
        uno = np.matmul(self.coefficients[2], c)
        dos = (uno.T * c.T).sum(-1)
        tres = np.dot(c.T, self.coefficients[1])
        # # print(tres.shape)
        return self.coefficients[0] + tres + dos

    def get_prev(self, matrix: bool = False) -> tuple[list[float], list[float]] \
                                                | list[float]:
        """
        Fetches previous layers neurons(x1 and x2) outputs

        :param prev: previous layer
        :param matrix: if True uses matrices for calculating previous layers output
        :return: results of the neurons that are to be combined
        """
        if self.first:
            if self.through:
                return self.prev[:, self.x1]
            x1 = self.prev[:, self.x1]
            x2 = self.prev[:, self.x2]
        else:
            if matrix:
                if self.through:
                    return self.prev[self.x1].calc_quadratic_matrix()
                x1 = self.prev[self.x1].calc_quadratic_matrix()
                x2 = self.prev[self.x2].calc_quadratic_matrix()
            else:
                if self.through:
                    return self.prev[self.x1].calc_quadratic()
                x1 = self.prev[self.x1].calc_quadratic()
                x2 = self.prev[self.x2].calc_quadratic()
        return x1, x2

    # storage of previous input vs calculating it every time...
    def regression_of_function(self, input_x1: np.ndarray, input_x2: np.ndarray | None, y: np.ndarray,
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
        return self.get_error(input_x1, input_x2, y, fitness_fn, tip=True)

    def get_error(self, input_x1: np.ndarray, input_x2: np.ndarray | None, y: np.ndarray,
                  fitness_fn: Callable = least_squares_error, tip: bool = True) -> float:
        """
        Calculates error between the prediction and ground truth based on passed function
        :param tip: wether this is training or testing error
        :param input_x1: previous layers first neurons output
        :param input_x2:  previous layers second neurons output or None if it's a pass through layer
        :param y: ground truth for predicted variable
        :param fitness_fn: function to use for calculating the error has to have 1st argument be the predicted values
        and second argument the ground truth
        :return: calculated error between prediction and ground truth
        """
        if tip:
            if self.results[0] is None:
                self.results[0] = self.calc_quadratic_matrix()
                if len(self.results[0]) > len(input_x1):
                    self.results[0] = self.results[0][:len(input_x1)]
            res = self.results[0]
        else:
            if self.results[1] is None:
                self.results[1] = self.calc_quadratic_matrix()
                if len(self.results[1]) > len(input_x1):
                    self.results[1] = self.results[1][-len(input_x1):]
            res = self.results[1]
        # res = self.calc_quadratic_matrix(input_x1, input_x2)
        # print(len(res))
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
            input_x1, input_x2 = self.get_prev()
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            train_x2 = input_x2[:floor(len(input_x2) * split)]
            train_y = y[:floor(len(y) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = input_x2[floor(len(input_x2) * split):]
            test_y = y[floor(len(y) * split):]
            train_res = self.regression_of_function(train_x1, train_x2, train_y, fitness_fn=fitness_fn)
        else:
            input_x1 = self.get_prev()
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = None
            train_y = y[:floor(len(y) * split)]
            test_y = y[floor(len(y) * split):]
            # print(len(train_x1), len(train_y))
            train_res = self.get_error(train_x1, None, train_y, fitness_fn, tip=True)

        selection_res = self.get_error(test_x1, test_x2, test_y, fitness_fn, tip=False)
        return train_res, selection_res

    def calc_quadratic_forward(self, x1, x2):
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
                x1 ** 2) + self.c_non_matrix[4] * (x2 ** 2) + self.c_non_matrix[5] * x1 * x2 for x1, x2 in zip(x1, x2)]

    def calc_quadratic_matrix_forward(self, x1, x2) -> np.ndarray:
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
        uno = np.matmul(self.coefficients[2], c)
        dos = (uno.T * c.T).sum(-1)
        tres = np.dot(c.T, self.coefficients[1])
        # # print(tres.shape)
        return self.coefficients[0] + tres + dos

    def forward_pass(self, prev_layer_res):
        if type(prev_layer_res) is np.ndarray:
            if len(prev_layer_res.shape) == 1:
                x1 = prev_layer_res[self.x1]
                x2 = prev_layer_res[self.x2] if not self.through else None
            else:
                x1 = prev_layer_res[:, self.x1]
                x2 = prev_layer_res[:, self.x2] if not self.through else None
        else:
            x1 = prev_layer_res[self.x1]
            x2 = prev_layer_res[self.x2] if not self.through else None
        return self.calc_quadratic_matrix_forward(x1, x2)
