import numpy as np
from itertools import combinations as comb
import concurrent.futures as ft
from typing import Callable, Union
from math import floor
from random import sample
import dill
import warnings
#warnings.filterwarnings("error")

# use scipy for value estimation
from scipy.optimize import least_squares


class DimMismatch(Exception):

    def __init__(self, shape, message=f"First dimension is not 2"):
        super().__init__(message)
        self.shape = shape

    def __str__(self):
        return f"The shape of the matrix is {self.shape}, first dimension is not 2"


class ShapeMismatchException(Exception):
    def __init__(self, shape, message="shape has more than 2 dimensions"):
        super().__init__(message)
        self.message = message
        self.shape = shape

    def __str__(self):
        return f"Array is of shape: {self.shape} -> {self.message}"


class ModelNotTrained(Exception):
    def __init__(self, message="Models coefficients have yet to be computed"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


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

    def __init__(self, inputs: np.ndarray, y: np.ndarray, max_neurons_per_layer: int = 128,
                 err_fn: Callable = least_squares_error, err_leeway: int = 1, split_train_select: float = 0.5) -> None:
        """
        Initializes the GMDH model per specified parameters and sets it into training mode(can not evaluate using it
        untill you train it

        :param inputs: input variables in a numpy array
        :param y: ground truth variable(forecasting target) in numpy array
        :param max_neurons_per_layer: max amount of neurons per layer, you might want to play with this setting if you
        wish to speed up training. Neurons are cut by random sampling from all combinations befor even being trained,
        this setting is mostly used to save up on space on lower end machines.
        :param err_fn: function for computing the error
        :param err_leeway: how many layers can underperform before we stop training
        :param split_train_select: ratio between train and selection set
        """
        self.layers = []
        self.inputs = inputs
        self.y = y
        self.max_neurons = max_neurons_per_layer
        self.err_fn = err_fn
        self.err_leeway = err_leeway
        self.split_train_select = split_train_select
        self.can_evaluate = False

    def train(self) -> None:
        """
        Trains the model
        """
        loss_cnt = 0
        print(f"Starting training")
        cur_layer_res = self.inputs
        self.layers.append(GMDHLayer(self.inputs))
        min_loss = self.layers[-1].train_layer(prev=self.inputs, y=self.y, fitness_fn=self.err_fn,
                                               split=self.split_train_select)
        cur_layer_res = self.layers[-1].forward(cur_layer_res)
        print(f"Layer 0 Trained: {len(self.layers[-1])}")
        while True:
            print(f"Training layer {len(self.layers)}")
            self.layers.append(GMDHLayer(inputs=cur_layer_res))
            cur_loss = self.layers[-1].train_layer(prev=cur_layer_res, y=self.y, fitness_fn=self.err_fn,
                                                   split=self.split_train_select)
            cur_layer_res = self.layers[-1].forward(cur_layer_res)
            print(f"Layer {len(self.layers) - 1} Trained: {len(self.layers[-1])}")
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

    def test(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluates the model on given input

        :param inputs: numpy array of input variables
        :return:
        """
        if not self.can_evaluate:
            print("Model can not be evaluated yet, please first train the model")
            return -1
        if type(inputs) is not np.ndarray:
            raise TypeError(f"Inputs type {type(inputs)} is not a numpy array")

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs


# fixed
class GMDHLayer:

    # fixed
    def __init__(self, inputs: np.ndarray, threshold: float = None, parallel: bool = False, workers: int = 1,
                 max_neurons: int = 128) -> None:
        """
        initializes the layer

        :param inputs: collection of output variables in a np.ndarray from previous layer
        :param threshold: a threshold to get rid of neurons that are not good enough, so we don't waste our time with
        comparing and space for storing, if not set it's set to median of previous layer
        :param parallel: do we want to parallelize the workload
        :param workers: number of workers to use in parallelization
        :param max_neurons: max amount of neurons in layer
        """
        # self.index = index
        # print(f"GMDHLayer: {inputs}")
        self.max_neurons = max_neurons
        self.neurons = [(-1, PolyLeastSquares(inp)) for inp in comb(range(inputs.shape[1]), 2)]

        if len(self.neurons) > 1000:
            sample(self.neurons, max_neurons)
        # print("oop")
        self.parallel = parallel
        self.workers = workers
        self.threshold = threshold
        self.frozen = False
        # self.prev_layer = inputs

    # fixed
    def __len__(self):
        return len(self.neurons)

    # fixed
    def __getitem__(self, i):
        return self.neurons[i][1]

    # fixed
    def train_layer(self, prev: np.ndarray, y: np.ndarray,
                    fitness_fn: Callable = least_squares_error,
                    split: float = 0.5) -> int:
        """
        Trains layer and selects, which neurons to keep from previous layers, which to replace and which new neurons
        to add

        :param prev: previous layer output in numpy array
        :param y: numpy array of ground truths
        :param fitness_fn: fitness function to be used for error calculation
        :param split: ratio between training and selection sets
        :return: the minimum error of all the neurons
        """
        if self.frozen:
            print("layer is frozen it can not be trained")
            return -1

        # entries are tuples of (fitness and neurons)
        accepted_comp = []
        if self.parallel:
            pass
            with ft.ProcessPoolExecutor(max_workers=self.workers) as executor:
                futs = [executor.submit(neuron[1].regress_and_test(prev, y)) for neuron in self.neurons]
                ft.wait(futs)
        else:
            # creation and addition of previous layers neurons as pass-through neurons
            for neuron in range(prev.shape[1]):
                neuro = PolyLeastSquares(neuron, through=True, )
                # print(split)
                err = neuro.regress_and_test(prev, y, fitness_fn, split)
                accepted_comp.append((err[1], neuro))
            accepted_comp.sort()
            # print(f"first: {neuro}")
            threshold = accepted_comp[floor(len(accepted_comp) / 2)][0]

            # test new neurons and add them to queue if condition is satisfied
            while len(self.neurons) > 0:
                neuron = self.neurons.pop()
                # print(f"uno neuron{neuron}")
                error = neuron[1].regress_and_test(prev, y, fitness_fn, split)
                # print(threshold, error)
                if error[1] < threshold:
                    # print(f"YOU ARE HIRED")
                    accepted_comp.append((error[1], neuron[-1]))
            # print(f"Accepted trained:{accepted_comp[0]},{accepted_comp[1]}")
            accepted_comp.sort()
            self.neurons = accepted_comp
            # print(self.neurons)

            return self.neurons[0][0]

    # fixed
    def reduce_to_best_n_neuron(self, n: int = 1) -> None:
        """
        Reduces the layer to n best performing neurons, usually used with n=1 for the last layer of a model

        :param n: number of neurons to keep
        """
        self.neurons = [self.neurons[:n]]

    # fixed
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward passes the input through the layer

        :param inputs: numpy array of previous layers results or the initial input if this is the first layer
        :return: results of this layer stacked in a numpy array
        """
        # print(self.neurons)
        res = []
        print(self.neurons)
        for n in self.neurons:
            res.append(n[1].forward_pass(inputs))
        return np.stack(res)

    def freeze_layer(self) -> None:
        """
        Freezes the layer to prevent updating or retraining the neurons
        """
        if self.frozen:
            print("Layer already frozen")
        else:
            self.frozen = True

    def unfreeze(self) -> None:
        """
        Unfreezes the layer to allow for training or updating weights
        """
        if not self.frozen:
            print("Layer already unfrozen")
        else:
            self.frozen = False


class PolyLeastSquares:

    # fixed
    def __init__(self, input_indexes: list[int] | int, coefficients: list[float] = None,
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
        # self.prev_results = prev_results
        if through:
            self.x1 = input_indexes
            self.x2 = None
        else:
            self.coefficients = None if not coefficients else self.to_lin_alg(coefficients)
            self.x1, self.x2 = input_indexes
        # print(f"New wave{first}")

    def __str__(self) -> str:
        """
        Gvies short summary of the object

        :return: summary of object
        """
        return f"""
Neuron specification:
    Indexes: x1={self.x1} x2={self.x2}
    Coefficients: {self.coefficients}
        """

    # fixed
    @staticmethod
    def to_lin_alg(c):
        """
        converts coefficient into matrix form

        :param c:
        :return:
        """
        return c[0], np.array([c[1], c[2]]), np.array([[c[3], c[5] / 2], [c[5] / 2, c[4]]])

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
            raise ModelNotTrained()
        if type(x1) is not np.ndarray:
            raise TypeError(f"First inputs type {type(x1)} is not a numpy array")
        if type(x2) is not np.ndarray and x2 is not None:
            raise TypeError(f"Second inputs type {type(x2)} is not a numpy array and is not None")
        if len(x1.shape) > 3:
            raise ShapeMismatchException(x1.shape)
        if x2 is not None and len(x2.shape) > 3:
            raise ShapeMismatchException(x2.shape)
        c = np.array([x1, x2])
        if c.shape[0] != 2:
            raise DimMismatch(c.shape)

        # print(c.shape,)
        # Z = N.diag(X.dot(Y)) -> Z = (X * Y.T).sum(-1)
        uno = np.matmul(self.coefficients[2], c)
        dos = (uno.T * c.T).sum(-1)
        tres = np.dot(c.T, self.coefficients[1])
        # # print(tres.shape)
        return self.coefficients[0] + tres + dos

    # fixed
    def get_prev(self, prev: np.ndarray) -> tuple[list[float], list[float]] | list[float]:
        """
        Fetches previous layers neurons(x1 and x2) outputs

        :return: results of the neurons that are to be combined
        """
        if self.through:
            return prev[:, self.x1], None
        x1 = prev[:, self.x1]
        x2 = prev[:, self.x2]
        return x1, x2

    # fixed
    def regression_of_function(self, input_x1: np.ndarray, input_x2: np.ndarray | None, y: np.ndarray,
                               fitness_fn: Callable = least_squares_error) -> float:
        """
        Runs least squares regression to solve for A,B,C,D,E,F in:
         A + B * u + C * v + D * u^2 + E * v^2 + F * u * v

        :param input_x1: first inputs results
        :param input_x2: second inputs results can be None if it's a pass through neuron
        :param fitness_fn: function for calculating the error
        :param y: target variable
        :return: returns error of the function
        """

        if type(input_x1) is not np.ndarray:
            raise TypeError(f"First inputs type {type(input_x1)} is not a numpy array")
        if type(input_x2) is not np.ndarray and input_x2 is not None:
            raise TypeError(f"Second inputs type {type(input_x2)} is not a numpy array and is not None")
        if len(input_x1.shape) > 3:
            raise ShapeMismatchException(input_x1.shape)
        if input_x2 is not None and len(input_x2.shape) > 3:
            raise ShapeMismatchException(input_x2.shape)

        # c = np.array([prev[self.x1],prev[self.x2]])
        A = np.array([input_x1 * 0 + 1, input_x1, input_x2, input_x1 ** 2, input_x2 ** 2, input_x1 * input_x2]).T

        # euclidean 2-norm based residual
        coeff, r, rank, s = np.linalg.lstsq(A, y)
        self.coefficients = self.to_lin_alg(coeff)

        # calculating fitness
        return self.get_error(input_x1, input_x2, y, fitness_fn)

    # fixed
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
        if type(y) is not np.ndarray:
            raise TypeError(f"Ground truth type {type(y)} is not a numpy array")
        if type(input_x1) is not np.ndarray:
            raise TypeError(f"First inputs type {type(input_x1)} is not a numpy array")
        if type(input_x2) is not np.ndarray and input_x2 is not None:
            raise TypeError(f"Second inputs type {type(input_x2)} is not a numpy array and is not None")
        if len(input_x1.shape) > 3:
            raise ShapeMismatchException(input_x1.shape)
        if input_x2 is not None and len(input_x2.shape) > 3:
            raise ShapeMismatchException(input_x2.shape)

        res = self.calc_quadratic_matrix(input_x1, input_x2)
        return fitness_fn(res, y)

    # fixed
    def regress_and_test(self, prev: np.ndarray, y: np.ndarray, fitness_fn: Callable = least_squares_error,
                         split: float = 0.5) -> tuple[float, float]:
        """
        Regresses the neuron and calculates both train and selection error for said neuron

        :param prev: complete previous layers output in case this is first layer the previous layer is inputs
        :param y: ground truth values to be predicted
        :param fitness_fn: function for calculating the error between predicted and ground truth values
        :param split: ratio between train and selection set
        :return: error on train set, error on selection set(selection set is the one that matters)
        """
        if type(prev) is not np.ndarray:
            raise TypeError(f"Prev layers results type {type(prev)} is not a numpy array")
        if type(y) is not np.ndarray:
            raise TypeError(f"y type {type(prev)} is not a numpy array")
        if split >= 1 or split <= 0:
            raise ValueError(f"split is {split} -> constraint 0 <= split <= 1 not satisfied")

        # print("still have it", prev.shape)
        if not self.through:
            input_x1, input_x2 = self.get_prev(prev)
            del prev
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            train_x2 = input_x2[:floor(len(input_x2) * split)]
            train_y = y[:floor(len(y) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = input_x2[floor(len(input_x2) * split):]
            test_y = y[floor(len(y) * split):]
            train_res = self.regression_of_function(train_x1, train_x2, train_y, fitness_fn=fitness_fn)
        else:
            input_x1, _ = self.get_prev(prev)
            del prev
            train_x1 = input_x1[:floor(len(input_x1) * split)]
            test_x1 = input_x1[floor(len(input_x1) * split):]
            test_x2 = None
            train_y = y[:floor(len(y) * split)]
            test_y = y[floor(len(y) * split):]
            train_res = self.get_error(train_x1, None, train_y, fitness_fn)
        selection_res = self.get_error(test_x1, test_x2, test_y, fitness_fn)
        return train_res, selection_res

    # fixed
    def forward_pass(self, prev_layer_res: np.ndarray) -> np.ndarray:
        """
        Passes through previous layers output as input and returns this layers output

        :param prev_layer_res: results from the previous layer in a numpy array
        :return: neurons results
        """
        if type(prev_layer_res) is not np.ndarray:
            raise TypeError(f"{type(prev_layer_res)} is not a numpy array")
        if len(prev_layer_res.shape) == 1:
            x1 = prev_layer_res[self.x1]
            x2 = prev_layer_res[self.x2] if not self.through else None
        elif len(prev_layer_res.shape) == 2:
            x1 = prev_layer_res[:, self.x1]
            x2 = prev_layer_res[:, self.x2] if not self.through else None
        else:
            raise ShapeMismatchException(prev_layer_res.shape)
        del prev_layer_res
        return self.calc_quadratic_matrix(x1, x2)


class DataLoader:

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def window(self, window_size: int = 30) -> np.ndarray:
        return np.lib.stride_tricks.sliding_window_view(self.data, window_shape=window_size)

    def window_split_x_y(self, window_size: int = 30, y_len: int = 1):
        x = self.window(window_size=window_size)
        return x[:, :-y_len], x[:, -y_len:]

    def window_split_train_val_x_y(self, train_val_split: float = 0.8, window_size: int = 30, y_len: int = 1):
        x, y = self.window_split_x_y(window_size=window_size, y_len=y_len)

        train_x, val_x = x[:floor(len(x) * train_val_split), :], x[floor(len(x) * train_val_split):, :]
        if len(y.shape) == 1:
            train_y, val_y = y[:floor(len(y) * train_val_split)], y[floor(len(y) * train_val_split):]
        else:
            train_y, val_y = y[:floor(len(y) * train_val_split), :], y[floor(len(y) * train_val_split):, :]

        return (train_x, train_y), (val_x, val_y)

    def window_split_train_select_val_x_y(self, train_select_split: float = 0.5, train_val_split: float = 0.8,
                                          window_size: int = 30, y_len: int = 1):
        x, y = self.window_split_x_y(window_size=window_size, y_len=y_len)

        # TODO fix from here onwards so that train_x doesn't bleed into val_x with the features(window size)
        train_x, val_x = x[:floor(len(x) * train_val_split)-window_size, :], x[floor(len(x) * train_val_split):, :]
        if len(y.shape) == 1:
            train_y, val_y = y[:floor(len(y) * train_val_split)-window_size], y[floor(len(y) * train_val_split):]
        else:
            train_y, val_y = y[:floor(len(y) * train_val_split)-window_size, :], y[floor(len(y) * train_val_split):, :]

        train_x, select_x = train_x[:floor(len(train_x) * train_select_split)-window_size, :], \
                            train_x[floor(len(train_x) * train_select_split):, :]
        if len(train_y.shape) == 1:
            train_y, select_y = train_y[:floor(len(train_y) * train_select_split)-window_size], train_y[floor(
                len(train_y) * train_select_split):]
        else:
            train_y, select_y = train_y[:floor(len(train_y) * train_select_split)-window_size, :], train_y[floor(
                len(train_y) * train_select_split):, :]
        print(f"Creating my dataload: "
            f"val beginning = {floor(len(x) * train_val_split)},"
            f"test beginning = {floor(len(train_x) * train_select_split)}")
        return (train_x, train_y), (select_x, select_y), (val_x, val_y)

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100,
                     fill = '█', printEnd = ''):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def create_combs(X: np.ndarray):
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            yield X[:, i], X[:, j]
    return None


def poly(coeff: np.ndarray | list, x: np.ndarray, use_poly=True):
    # pass through
    if not use_poly:
        return x

    coeffs = coeff[0], np.array([coeff[1], coeff[2]]), np.array([[coeff[3], coeff[5] / 2], [coeff[5] / 2, coeff[4]]])
    if x.shape[0] != 2:
        raise DimMismatch(x.shape)
    # print(x.shape,)
    # Z = N.diag(X.dot(Y)) -> Z = (X * Y.T).sum(-1)
    uno = np.matmul(coeffs[2], x)
    dos = (uno.T * x.T).sum(-1)
    tres = np.dot(x.T, coeffs[1])
    # # print(tres.shape)
    res = coeffs[0] + tres + dos
    #print(f"Last line of defence: {x.shape}, {res.shape}")
    return res


def radial_basis(coeffs, x, use_poly = True):
    if use_poly:
        return np.exp(-np.square(poly(coeffs, x)))
    else:
        return np.exp(-np.square(x))


def inverse_radial_basis(y):
    try:
        return np.sqrt(-np.log(y))
    except RuntimeWarning:
        print(np.log(y))
        print(y, np.sqrt(-np.log(y)))


def sigmoid(coeffs, x, use_poly = True):
    if use_poly:
        return 1/(1+np.exp(-poly(coeffs,x)))
    else:
        return 1 / (1 + np.exp(-x))


def inverse_sigmoid(y):
    return np.log(y / (1 - y))


def hyperbolic_tangent(coeffs, x, use_poly = True):
    if use_poly:
        return 2 / (1 + np.exp(-2 * poly(coeffs, x)))
    else:
        return 2 / (1 + np.exp(-2 * x))


def inverse_hyperbolic_tangent(y):
    #print(2 / (y + 1) - 1,np.log((2 / (y + 1) - 1)) )
    return -np.log((2 / (y + 1) - 1))/2


def normalize_ts(ts, ratio):
    mi = min(ts[:floor(len(ts)*ratio)])
    mx = max(ts[:floor(len(ts)*ratio)])
    return (ts - mi)/(mx - mi)


class GMDHSlim:

    def __init__(self, transfer_functions: list[tuple[Callable, Callable]] | tuple[tuple[Callable, Callable]] = [],
                 error_function: Callable = mean_square_error,
                 train_select_split: float = 0.75, max_layer_size: int = 128):
        self.transfer_functions = transfer_functions
        self.error_function = error_function
        self.ts_split = train_select_split
        self.layers = []
        self.max_layer_size = max_layer_size

    def model_composition(self):
        for i, l in enumerate(self.layers):
            print(f"Layer {i}: {len(l)}")

    # DONE
    def construct_GMDH(self, X_train, y_train, X_select, y_select, stop_leniency):
        cost = np.inf
        cost_bad = 0
        i = 1
        print("Starting training of neural network")
        while 1:
            print(f"Training layer {i}, current shapes:{X_train.shape}, {X_select.shape}")
            self.layers.append(MatrixGMDHLayer(self.transfer_functions, self.error_function, self.ts_split,
                                               self.max_layer_size))
            cur_best_neur = self.layers[-1].train_layer(X_train, y_train, X_select, y_select)
            X_train = self.layers[-1].forward(X_train)
            X_select = self.layers[-1].forward(X_select)
            #print(f"In construct GMDH after one pass{X_train.shape}, {X_select.shape}")
            print(f"Layer {i} trained")

            if cost > cur_best_neur[0]:
                cost = cur_best_neur[0]
                cost_bad = 0
            else:
                print(f"Layer {i} performed worse than previous layer")
                cost_bad = cost_bad + 1
                if cost_bad == stop_leniency:
                    print(f"Amount of worse performances has exceeded {stop_leniency} finishing training of the model!")
                    self.layers = self.layers[:-stop_leniency]
                    self.layers[-1].reduce_to_output()
                    break
            i = i + 1

    def evaluate(self, X):
        #print(X.shape)
        for i in self.layers:
            X = i.forward(X)
        #print(X)
        return X

class MatrixGMDHLayer:
    """
    THIS CLASS USES THE TRANSFER FUNCTION AS INPUT NOT OUTPUT
    INPUT -> TRANSFER FUNCTION -> POLY -> INVERSE TRANSFER FUNCTION -> OUTPUT
    """
    def __init__(self, transfer_functions: list = [], error_function: Callable = mean_square_error,
                 train_select_split: float = 0.75, max_layer_size: int = 128):
        self.transfer_functions = transfer_functions
        self.error_function = error_function
        self.added_value = (0, 0)
        self.ts_split = train_select_split
        self.layer = None
        self.max_layer_size = max_layer_size

    def __len__(self):
        return len(self.layer)

    @staticmethod
    def calc_poly_coeff(X: np.ndarray, y: np.ndarray, transfer_func: Callable):
        """
        Calculates and chooses the best

        :param X: shape = (2 combined input variables, length of sample)
        :param y: shape = (length of sample)
        :param transfer_func: transfer function that accepts coeff, X
        :return: coeff with the highest value
        """
        # removed the transformation on ts_y
        #print(f"We going into poly territory: {X.shape}, {y.shape}")
        res = least_squares(lambda coeffs, ts_x, ts_y, ts_f: np.squeeze(ts_y) - poly(coeffs, ts_x),
                            np.array([3, 1, 2, 1, 1, 1]), args=(X, y, transfer_func))
        return res.x, res.cost

    @staticmethod
    def evaluate_poly(coeffs, X: np.ndarray, y: np.ndarray, transfer_fn: tuple[Callable, Callable],
                      cost_fn: Callable = mean_square_error):
        return cost_fn(poly(coeffs, X), y)
        #return cost_fn(transfer_fn[1](poly(coeffs, X)), transfer_fn[1](y))

    def pick_best_combination_fn(self, X: np.ndarray, y: np.ndarray, transfer_fn: Callable, cost_fn: Callable):
        """
        Picks only the best performing combination and returns it

        :param X: input variables matrix of shape = (sample length, number of input variables)
        :param y: ground truth variable of shape = (sample length)
        :param transfer_fn: transfer function that we want to fit
        :param cost_fn: function used to calculate the cost/error of the trained result on the selection set
        :return: None
        """
        min_cost = np.inf
        best_performer = (-1, -1)
        best_coeff = None
        for i in comb(range(X.shape[1]), 2):
            train_matrix_x = np.array([X[:floor(X.shape[0] * self.ts_split), i[0]],
                                       X[:floor(X.shape[0] * self.ts_split), i[1]]])
            train_matrix_y = np.array(y[:floor(y.shape[0] * self.ts_split)])
            test_matrix_x = np.array([X[floor(X.shape[0] * self.ts_split):, i[0]],
                                      X[floor(X.shape[0] * self.ts_split):, i[1]]])
            test_matrix_y = np.array(y[floor(y.shape[0] * self.ts_split):])
            res = self.calc_poly_coeff(train_matrix_x, train_matrix_y, transfer_fn)
            coeffs = res[0]
            mse = self.evaluate_poly(coeffs, test_matrix_x, test_matrix_y, transfer_fn, cost_fn)
            mse_poly = self.evaluate_poly(coeffs, test_matrix_x, test_matrix_y, poly, cost_fn)
            if mse < min_cost:
                min_cost = mse
                best_performer = i
                best_coeff = coeffs
        self.new_combs = best_performer
        self.coeffs = best_coeff
        #print(self.indexes, " ", self.coeffs)
        return best_performer, coeffs, mse

    # DONE
    def train_combinations_tf(self, X_train: np.ndarray, y_train: np.ndarray, X_select: np.ndarray,
                              y_select: np.ndarray, transfer_fn: tuple[Callable, Callable],
                              cost_fn: Callable):
        """
        Picks only the best performing combination and returns it

        :param X_train: input variables matrix of shape = (sample length * train_split, number of input variables)
        :param X_select: input variables matrix of shape = (sample length * (1-train_split), number of input variables)
        :param y_train: ground truth variable of shape = (sample length * (train_split))
        :param y_select: ground truth variable of shape = (sample length * (1 - train_split))
        :param transfer_fn: transfer function that we want to fit and its inverse
        :param cost_fn: function used to calculate the cost/error of the trained result on the selection set
        :return: None
        """
        combs = []
        cntr = 0
        overall = (X_train.shape[1]*(X_train.shape[1]-1))/2
        X_train = transfer_fn[0](None, X_train, False)
        X_select = transfer_fn[0](None, X_select, False)
        #print(f"Train combinations({transfer_fn[0].__name__}): {X_train.shape}, {X_select.shape}")
        #y = transfer_fn[0](None, y, False)
        printProgressBar(0, overall, prefix=transfer_fn[0].__name__, suffix=str(overall))
        for cntr, i in enumerate(comb(range(X_train.shape[1]), 2)):
            printProgressBar(cntr+1, overall, prefix=transfer_fn[0].__name__, suffix=str(overall))
            train_matrix_x = np.array([X_train[:, i[0]], X_train[:, i[1]]])
            test_matrix_x = np.array([X_select[:, i[0]], X_select[:, i[1]]])

            #print(f"Combinations: {train_matrix_x.shape}, {test_matrix_x.shape}")
            res = self.calc_poly_coeff(train_matrix_x, y_train, poly)
            coeffs = res[0]

            mse_poly = self.evaluate_poly(coeffs, test_matrix_x, y_select, transfer_fn, cost_fn)
            combs.append((mse_poly, i, coeffs, transfer_fn))
            cntr = cntr + 1
        return combs

    # DONE
    def train_layer(self, X_train: np.ndarray, y_train: np.ndarray, X_select: np.ndarray, y_select: np.ndarray,
                    transfer_functions: list[tuple[Callable, Callable]] = None,
                    ensamble_function: Callable = None, cost_function: Callable = None,
                    select_function: Callable = lambda res: [min(range(len(res)), key=res.__getitem__)],
                    use_ensamble: bool = False) -> tuple[float, tuple[int, int], list[int], tuple[Callable, Callable]]:
        """
        Construct a train ensamble of n*comb(X.shape[1],2) where n is number of transfer functions given, and connect
        the different transfer functions, for the same combination using the ensamble rule. Evaluate the results using
        the cost function and select best results using the selection function

        :param X: np.array of shape = (sample length, number of input variables) input variables to train on
        :param y: np.array of shape = (sample length) ground truth
        :param transfer_functions: list of transfer functions and their inverses to be used for training the model,
        they have to recieve 3 parameters in this order: coefficients(this will be evaluated), input, bool check for
        wether the function is being used on the input variables(use the polynomial function) or the ground truth,
        if ground truth then set to False if input vars set to True.
        :param ensamble_function: function that receives np.array of shape = (combinations, number of transfer functions
        provided)
        :param cost_function: used for evaluating results for the selection process, input has to be (z: np.array,
        y: np.array) z being predicted result and y being the ground truth and returns the cost: float
        :param select_function: select_function receives as input np.array of shape =
        (floor(sample length * (1 - train_select_ratio)) of  returns a list of indexes
        :param use_ensamble: wether to construct an ensamble(True) or just chose best performing
        :return: best performing neuron
        """
        if transfer_functions is None:
            transfer_functions = self.transfer_functions
        if cost_function is None:
            cost_function = self.error_function
        costs_prev = []
        # get costs of prev layer
        for i in range(X_select.shape[1]):
            #costs_prev.append((cost_function(X[floor(X.shape[0] * self.ts_split):, i],
            #                                 y[floor(len(y) * self.ts_split):]), (i, -1), "TRANSFER"))
            costs_prev.append((cost_function(X_select[:, i], y_select), (i, -1), "TRANSFER"))
        costs_prev.sort()
        for tf in transfer_functions:
            #print(f"Layer train: {X_train.shape}, {X_select.shape}")
            costs_prev = costs_prev + self.train_combinations_tf(X_train, y_train, X_select, y_select, tf,
                                                                 cost_function)

        costs_prev.sort()
        self.layer = costs_prev[:self.max_layer_size]

        return costs_prev[0]
        #costs_prev

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        does a forward pass of the network

        :param X: input variables from previous layer
        :return:
        """
        if len(X.shape) > 2:
            X = np.squeeze(X, axis=0)
        res = []
        #n_layers = len(self.layer)
        try:
            for j, i in enumerate(self.layer):
                if i[1][1] == -1:

                    res.append(X[:, i[1][0]])
                else:
                    # use the neurons transfer function for the inputs
                    x1 = i[3][0](None, X[:, i[1][0]], False)
                    x2 = i[3][0](None, X[:, i[1][1]], False)
                    y = poly(i[2], np.array([x1, x2]))
                    res.append(y)
            res = np.array(res)
        except IndexError as e:
            print(e, ":", self.layer, X)

        #print(res)
        return res.T

    def reduce_to_output(self):
        self.layer = [self.layer[0]]

