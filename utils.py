import numpy as np
from itertools import combinations as comb


class PolyLeastSquares:

    def __init__(self, input_indexes, coefficients = None, first = False):
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

    def calc_quadratic(self, prev):
        """
        calculate this neurons output based on previous layers input

        :param prev: previous layers neurons
        :return: result of the quadratic function
        """
        if self.c_non_matrix is None:
            return -1
        if self.first:
            x1 = prev[self.x1]
            x2 = prev[self.x2]
        else:
            x1 = prev[self.x1].calc_quadratic()
            x2 = prev[self.x2].calc_quadratic()

        return self.c_non_matrix[0] + self.c_non_matrix[1]*x1 + self.c_non_matrix[2]*x2 + self.c_non_matrix[3]*(x1**2) +\
               self.c_non_matrix[4]*(x2**2) + self.c_non_matrix[5]*x1*x2

    def calc_quadratic_matrix(self, prev):
        """
        calculate this neurons output based on previous layers input

        :param prev: previous layers neurons
        :return: result of the quadratic function
        """
        if self.coefficients is None:
            return -1

        if self.first:
            x1 = prev[:, self.x1]
            x2 = prev[:, self.x2]
        else:
            x1 = prev[self.x1].calc_quadratic_matrix()
            x2 = prev[self.x2].calc_quadratic_matrix()

        c = np.array([x1, x2])
        return self.coefficients[0]+np.dot(self.coefficients[1], c)+ np.dot(c.T,np.dot(self.coefficients[2], c))

    def get_prev(self,prev):
        if self.first:
            x1 = prev[:, self.x1]
            x2 = prev[:, self.x2]
        else:
            x1 = prev[self.x1].calc_quadratic_matrix()
            x2 = prev[self.x2].calc_quadratic_matrix()

        return x1, x2

    # storage of previous input vs calculating it every time...
    def regression_of_function(self, prev, y):
        input_x1, input_x2 = self.get_prev(prev)
        #c = np.array([prev[self.x1],prev[self.x2]])
        A = np.array([input_x1 * 0 + 1, input_x1, input_x2, input_x1 ** 2, input_x2 ** 2, input_x1 * input_x2]).T

        coeff, r, rank, s = np.linalg.lstsq(A, y)
        self.c_non_matrix = coeff
        self.coefficients = self.to_lin_alg(coeff)
        return 1


class GMDHLayer:

    def __init__(self, inputs, first_layer=False):
        """
        initializes the layer

        :param inputs: previous layer nodes or original input
        """

        self.neurons = [PolyLeastSquares(inputs, first=first_layer) for inputs in comb(range(len(inputs)),2)]

