import numpy as np
from itertools import combinations as comb
import concurrent.futures as ft
from typing import Callable



class GMDHLayer:

    def __init__(self, inputs, threshold = None, first_layer=False, parallel = False, workers = 1):
        """
        initializes the layer

        :param inputs: previous layer nodes or original input, where previous layer is either GMDHLayer object or
        collection of input variables in a np.array
        :param threshold: a threshold to get rid of neurons that are not good enough, so we don't waste our time with
        comparing and space for storing, if not set its set to median of previous layer
        :param first_layer: check if its first layer since that would mean we don't get a previous layer but numpy array
        :param parallel: do we want to parallelize the workload
        :param workers: number of workers to use in parallelization
        """

        self.neurons = [PolyLeastSquares(inputs, first=first_layer) for inputs in (comb(range(len(inputs)), 2) if
                        not first_layer else comb(range(inputs.shape[1], 2)))]
        self.parallel = parallel
        self.workers = workers
        self.threshold = threshold
        self.prev_layer = inputs

    def __getitem__(self, i):
        return self.neurons[i]

    # this will probably be done in fitting process in the neuron
    def check_layer_fit(self, x, y):
        pass


    def train_layer(self,prev, y):
        # entries are tuples of (fitness and neurons)
        accepted_comp = []
        if self.parallel:
            with ft.ProcessPoolExecutor(max_workers=self.workers) as executor:
                futs = [executor.submit(neuron.regression_of_function(prev, y)) for neuron in self.neurons]
                ft.wait(futs)
        else:
            while len(self.neurons) > 0:
                neuron = self.neurons.pop()
                neuron.regression_of_function(prev, y)


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
            self.x1 , self.x2 = input_indexes
            self.first = first

    @staticmethod
    def to_lin_alg(c):
        """
        converts coefficient into matrix form

        :param c:
        :return:
        """
        return c[0], np.array([c[1],c[2]]), np.array([[c[3],c[5]/2],[c[5]/2, c[4]]])

    def calc_quadratic(self, prev: np.ndarray | GMDHLayer) -> list[float]:
        """
        calculate this neurons output based on previous layers input

        :param prev: previous layer
        :return: result of the quadratic function
        """
        if self.through:
            return self.get_prev(prev)
        if self.c_non_matrix is None:
            return -1
        x1, x2 = self.get_prev(prev)

        return [self.c_non_matrix[0] + self.c_non_matrix[1]*x1 + self.c_non_matrix[2]*x2 + self.c_non_matrix[3]*(x1**2) +\
               self.c_non_matrix[4]*(x2**2) + self.c_non_matrix[5]*x1*x2 for x1,x2 in zip(x1,x2)]

    def calc_quadratic_matrix(self, prev: np.ndarray | GMDHLayer) -> np.ndarray:
        """
        calculate this neurons output based on previous layers input

        :param prev: previous layers neurons
        :return: result of the quadratic function
        """
        if self.through:
            return self.get_prev(prev,matrix=True)
        if self.coefficients is None:
            return -1
        x1,x2 = self.get_prev(prev,matrix=True)

        c = np.array([x1, x2])
        return self.coefficients[0]+np.dot(self.coefficients[1], c)+ np.dot(c.T,np.dot(self.coefficients[2], c))

    def get_prev(self, prev: np.ndarray | GMDHLayer, matrix: bool = False) -> tuple[list[float],list[float]]:
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
    def regression_of_function(self, prev: np.ndarray | GMDHLayer, y: np.ndarray, fitness_fn: Callable = None) -> object:
        """
        Runs least squares regression to solve for A,B,C,D,E,F in:
         A + B * u + C * v + D * u^2 + E * v^2 + F * u * v

        :param fitness_fn:
        :param prev: previous layer
        :param y: target variable
        :return: returns 1 if successful (will be changed to loss in future)
        """
        input_x1, input_x2 = self.get_prev(prev)
        #c = np.array([prev[self.x1],prev[self.x2]])
        A = np.array([input_x1 * 0 + 1, input_x1, input_x2, input_x1 ** 2, input_x2 ** 2, input_x1 * input_x2]).T

        # euclidean 2-norm based residual
        coeff, r, rank, s = np.linalg.lstsq(A, y)
        self.c_non_matrix = coeff
        self.coefficients = self.to_lin_alg(coeff)

        # calculating fitness
        for x1,x2 in zip(input_x1,input_x2):
            self.calc_quadratic()

        return r



