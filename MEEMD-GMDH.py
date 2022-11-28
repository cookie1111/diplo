# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
from emd import sift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import PolyLeastSquares, GMDH, mean_square_error, DataLoader, MatrixGMDHLayer, radial_basis
import utils
from time import process_time_ns
from math import floor
from typing import Callable


TEST = 11


class MEEMDGMDH:

    def __init__(self, ts: np.ndarray) -> None:
        self.timeseries = ts
        self.models = None
        self.model_res = None

    def add_noise(self, noise_amp):
        return self.timeseries + np.random.normal(0, noise_amp, len(self.timeseries))

    def get_imfs(self, timeseries, upper_limit, cut_off = 0, amount_of_imfs: int = 9):
        s = sift.sift(timeseries[:upper_limit], max_imfs=amount_of_imfs)
        imfs = np.array(s[cut_off:, :])
        res = self.timeseries[cut_off:upper_limit] - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, procs: int = 1, noise_amp: float = 0.05,
                             use_split: float = 0.75, cut_off: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Calculate imfs for the given timeseries, and correct set(train/select/val) for each set the same start is
        used, just the cut-off point changes

        :param procs: amount of processes used CURRENTLY NOT IMPLEMENTED
        :param noise_amp: amplitude of the added gaussian noise
        :param use_split: add the upper limit -> whole set == 1, 0.5 means the upper limit is at hald of the set
        :param cut_off: starting position for the imfs(for example if its a select set this mean that we don't want to
        include the training part in the select)
        :return: imfs + res
        """
        if procs > 1:
            pass
        else:
            upper_limit = floor(len(self.timeseries)*use_split)
            noise_width = noise_amp * np.abs(np.max(self.timeseries[:upper_limit]) - np.min(self.timeseries[
                                                                                            :upper_limit]))
            number_ensamble_memebers = 1000
            all_imfs = {}
            all_res = []
            for i in range(number_ensamble_memebers):
                imf, res = self.get_imfs(self.add_noise(noise_width), upper_limit, cut_off=cut_off)
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

    def gmdh_train(self, train_x, train_y, select_x, select_y,
                   fitness_fn: tuple[tuple[Callable]] | list[tuple[Callable]] =
                   ((utils.poly, lambda x: x),
                    (utils.sigmoid, utils.inverse_sigmoid),
                    (utils.hyperbolic_tangent,utils.inverse_hyperbolic_tangent),
                    (utils.radial_basis, utils.inverse_radial_basis)),
                   err_function: Callable = mean_square_error,
                   split: float = 0.75):
        # print(train_x.shape, train_y.shape)
        model = utils.GMDHSlim(transfer_functions=fitness_fn,
                               error_function=err_function,
                               train_select_split=split)
        model.construct_GMDH(train_x, train_y, select_x, select_y, stop_leniency=3)
        return model

    # DONE
    def train_sets(self, fitness_fns, splits: tuple[float, float, float] = (0.75 * 0.8, 0.8, 1),) -> None:
        """
        Calculates the imfs for the ensambles and trains the corresponding models

        :param splits: ratio between train and selection set
        :return: None
        """
        self.models_imfs = []
        #train and selection
        imfs_train, res_train = self.create_median(*self.create_ensamble_imfs(use_split=splits[0]))

        imfs_select, res_select = self.create_median(
            *self.create_ensamble_imfs(use_split=splits[1], cut_off=floor(len(self.timeseries) * splits[0]))
        )
        # cut out only the relevant part
        imfs_test, res_test = self.create_median(
            *self.create_ensamble_imfs(use_split=splits[2], cut_off=floor(len(self.timeseries) * splits[1]))
        )
        print(f"Train:{imfs_train[0].shape}, Select:{imfs_select[0].shape}, Test:{imfs_test[0].shape}")

        for imf in zip(imfs_train, imfs_select, imfs_test):
            print("place_holder")
            dloader_train = DataLoader(imf[0])
            dloader_select = DataLoader(imf[1])
            dloader_test = DataLoader(imf[2])

            train_split = dloader_train.window_split_x_y(window_size=7)
            select_split = dloader_select.window_split_x_y(window_size=7)

            print(f"{train_split[0].shape} , {train_split[1].shape}")
            print(f"{select_split[0].shape} , {select_split[1].shape}")
            self.models[imf] = self.gmdh_train(*train_split, *select_split,
                                               fitness_fn=fitness_fns,
                                               err_function=mean_square_error)
        dloader_train = DataLoader(res_train)
        dloader_select = DataLoader(res_select)
        dloader_test = DataLoader(res_test)
        train_split = dloader_train.window_split_x_y(window_size=7)
        select_split = dloader_select.window_split_x_y(window_size=7)
        self.model_res = self.gmdh_train(*train_split, *select_split, mean_square_error)

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

    if TEST == 9:
        si = np.sin(0.1 * np.array(list(range(100)))) * 10
        ts = np.random.uniform(-1, 1, size=(100,))
        sig = utils.normalize_ts(si + ts, 0.75)
        plt.figure()
        plt.plot(range(len(ts)), sig)
        plt.show()
        matrix = np.lib.stride_tricks.sliding_window_view(sig, window_shape=7)
        gmdh = utils.GMDHSlim([(utils.poly, lambda x: x),
                               (utils.sigmoid, utils.inverse_sigmoid),
                               (utils.radial_basis, utils.inverse_radial_basis),
                               ],)
        gmdh.construct_GMDH(matrix[:, :-1], matrix[:, -1], 3)

    if TEST == 10:
        s = utils.normalize_ts(s, 0.75)
        matrix = np.lib.stride_tricks.sliding_window_view(s, window_shape=7)
        plt.figure()
        plt.plot(range(len(s)), s)
        plt.show()
        gmdh = utils.GMDHSlim([(utils.poly, lambda x: x),
                               (utils.sigmoid, utils.inverse_sigmoid),
                               (utils.hyperbolic_tangent,utils.inverse_hyperbolic_tangent),
                               (utils.radial_basis, utils.inverse_radial_basis)
                               ], )
        # (utils.radial_basis, utils.inverse_radial_basis)],)
        gmdh.construct_GMDH(matrix[:, :-1], matrix[:, -1], 3)

    if TEST == 11:
        s = utils.normalize_ts(s, 0.8)
        meme = MEEMDGMDH(s)
        meme.train_sets([(utils.poly, lambda x: x),
                         (utils.sigmoid, utils.inverse_sigmoid),
                         (utils.hyperbolic_tangent, utils.inverse_hyperbolic_tangent),
                         (utils.radial_basis, utils.inverse_radial_basis)
                         ],
                        )

