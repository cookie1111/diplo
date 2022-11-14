import numpy as np


class PolyLeastSquares:

    def __init__(self, input_indexes, coefficients, first = False):
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
        if not hasattr(self, "c_non_matrix"):
            return -1
        if self.first:
            x1 = prev[self.x1]
            x2 = prev[self.x2]
        else:
            x1 = prev[self.x1].calc_quadratic_matrix()
            x2 = prev[self.x2].calc_quadratic_matrix()

        return self.c_non_matrix[0] + self.c_non_matrix[1]*x1 + self.c_non_matrix[2]*x2 + self.c_non_matrix[3]*(x1**2) +\
               self.c_non_matrix[4]*(x2**2) + self.c_non_matrix[5]*x1*x2

    def calc_quadratic_matrix(self, prev):
        """
        calculate this neurons output based on previous layers input

        :param prev: previous layers neurons
        :return: result of the quadratic function
        """
        if not hasattr(self, "coefficients"):
            return -1
        if self.first:
            x1 = prev[self.x1]
            x2 = prev[self.x2]
        else:
            x1 = prev[self.x1].calc_quadratic_matrix()
            x2 = prev[self.x2].calc_quadratic_matrix()

        c = np.array([x1, x2])

        return self.coefficients[0]+np.dot(self.coefficients[1], c)+np.dot(c, np.dot(self.coefficients[2], c.T))

    # storage of previous input vs calculating it every time...
    def regression_of_function(self, input_x1, input_x2, Y):

        c = np.array([input_x1, input_x2])
        A = np.array([input_x1 * 0 + 1, input_x1, input_x2, input_x1 ** 2, input_x2 ** 2, input_x1 * input_x2]).T

        coeff, r, rank, s = np.linalg.lstsq(A,Y)
        self.c_non_matrix = coeff
        self.coefficients = self.to_lin_alg(coeff)
        return 1