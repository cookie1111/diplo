# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
from emd import sift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import PolyLeastSquares, GMDH, mean_square_error, DataLoader, MatrixGMDHLayer, radial_basis
import utils
from time import process_time_ns
from math import floor


TEST = 8

class MEEMDGMDH:

    def __init__(self, ts):
        self.timeseries = ts
        self.models = None
        self.model_res = None

    def add_noise(self, noise_amp):
        return self.timeseries + np.random.normal(0, noise_amp, len(self.timeseries))

    def get_imfs(self, timeseries):
        s = sift.sift(timeseries, max_imfs=9)
        imfs = np.array(s)
        res = self.timeseries - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, procs=1, noise_amp=0.05):
        if procs > 1:
            pass
        else:
            noise_width = noise_amp * np.abs(np.max(self.timeseries) - np.min(self.timeseries))
            number_ensamble_memebers = 1000
            all_imfs = {}
            all_res = []
            for i in range(number_ensamble_memebers):
                imf, res = self.get_imfs(self.add_noise(noise_width))
                for j in range(imf.shape[1]):
                    if j in all_imfs:
                        all_imfs[j].append(np.insert(imf[:, j], 0, i))
                    else:
                        all_imfs[j] = []
                        all_imfs[j].append(np.insert(imf[:, j], 0, i))
                    all_res.append(res)
            return all_imfs, all_res

    def create_median(self, imfs, res):
        imfs_medians = []
        for i in imfs:
            #print(len(imfs),len(imfs[0]),len(imfs[0][0]))
            nup = np.median(np.array(imfs[i]),axis=0)
            imfs_medians.append(nup)
        nup = np.median(np.array(res), axis=0)
        res_median = [nup]
        return imfs_medians, res_median

    def gmdh(self, train_x, train_y, fitness_fn, split):
        print(train_x.shape, train_y.shape)
        model = GMDH(train_x,train_y,err_fn=fitness_fn, split_train_select=split)
        model.train()
        return model
        #return indexs, coefficients

    def train(self, split: float = 0.5,) -> None:
        """
        Calculates the imfs for the ensambles and trains the corresponding models

        :param split: ratio between train and selection set
        :return: None
        """
        self.models_imfs = []
        imfs, res = self.create_ensamble_imfs()
        medians_imfs, median_res = self.create_median(imfs,res)
        for imf in medians_imfs:
            print("place_holder")
            dloader = DataLoader(imf)
            train_split, val_split = dloader.window_split_train_val_x_y(window_size=7)
            self.models[imf] = self.gmdh(*train_split, mean_square_error, split)
        dloader = DataLoader(res)
        train_split, val_split = dloader.window_split_train_val_x_y(window_size=7)
        self.model_res = self.gmdh(*train_split, mean_square_error, split)


    def test(self, inputs=None):
        if inputs is None:
            pass


if __name__ == '__main__':
    df = pd.read_csv("snp500.csv")
    p = PolyLeastSquares([0,1],[1,5,4,3,2,7])
    s = df['Close']
    ctr = 0
    if TEST == 1:
        # maybe good for parallel proccessing atm its useless doe
        start = process_time_ns()
        print(p)
        matrix = np.lib.stride_tricks.sliding_window_view(s,window_shape=7)
        c = p.calc_quadratic_matrix(matrix[:, 0], matrix[:, 1])
        m = process_time_ns() - start
        print(f"Speed quad matrix: {m}")
        start = process_time_ns()
        no_matrix = process_time_ns() - start
        print(f"Speed quad: {no_matrix}")
        print(f"matrix - no_matrix = {m-no_matrix}")
    if TEST == 2:
        matrix = np.lib.stride_tricks.sliding_window_view(s, window_shape=8)
        print(matrix[:, :7].shape, matrix[:, -1].shape)
        #p.regression_of_function(matrix[:, :7], matrix[:, -1:])
    if TEST == 3:
        matrix = np.lib.stride_tricks.sliding_window_view(s, window_shape=7)
        gmdh = GMDH(matrix[:, :-1], matrix[:, -1], err_fn=mean_square_error, max_neurons_per_layer=128)
        gmdh.train()
        print(gmdh.test(matrix[0, :-1]), matrix[0, -1])
    if TEST == 4:
        dl = DataLoader(s.iloc[:100])
        print(dl.window_split_x_y())
        print(dl.window_split_train_select_val_x_y())
    if TEST == 5:
        end_model = MEEMDGMDH(s)
        end_model.train()

    if TEST == 6:
        si = np.sin(0.1*np.array(list(range(100))))*10
        ts = np.random.uniform(-1, 1, size=(100,))
        sig = si + ts
        plt.figure()
        plt.plot(range(len(ts)), si + ts)
        plt.show()
        matrix = np.lib.stride_tricks.sliding_window_view(sig, window_shape=7)
        gmdh = MatrixGMDHLayer([radial_basis])
        insert = np.array([matrix[:, 0], matrix[:, 1]])
        gmdh.calc_poly_coeff(insert, matrix[:, -1])

    if TEST == 7:
        si = np.sin(0.1 * np.array(list(range(100)))) * 10
        ts = np.random.uniform(-1, 1, size=(100,))
        sig = si + ts
        plt.figure()
        plt.plot(range(len(ts)), si + ts)
        plt.show()
        matrix = np.lib.stride_tricks.sliding_window_view(sig, window_shape=7)
        gmdh = MatrixGMDHLayer([radial_basis])
        gmdh.pick_best_combination_fn(matrix[:, :-1], matrix[:, -1], radial_basis)

    if TEST == 8:
        si = np.sin(0.1 * np.array(list(range(100)))) * 10
        ts = np.random.uniform(-1, 1, size=(100,))
        sig = utils.normalize_ts(si + ts, 0.75)
        plt.figure()
        plt.plot(range(len(ts)), sig)
        plt.show()
        matrix = np.lib.stride_tricks.sliding_window_view(sig, window_shape=7)
        gmdh = MatrixGMDHLayer([radial_basis])
        gmdh.train_layer(matrix[:, :-1], matrix[:, -1], [(utils.poly, lambda x: x),
                                                         (utils.sigmoid, utils.inverse_sigmoid),
                                                         (utils.radial_basis, utils.inverse_radial_basis)],
                                                         #(utils.hyperbolic_tangent,utils.inverse_hyperbolic_tangent)],
                         ensamble_function=None)
        print(gmdh.forward(matrix[:, :-1]).shape)


