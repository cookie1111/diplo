# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
import os.path

from emd import sift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import PolyLeastSquares, GMDH, mean_square_error, DataLoader, MatrixGMDHLayer, radial_basis
import utils
from time import process_time_ns
from math import floor
from typing import Callable
#import pickle as pkl
import dill as pkl


TEST = 11


class MEEMDGMDH:

    def __init__(self, ts: np.ndarray, max_layer_size: int,  file_name: str) -> None:
        self.timeseries = ts
        self.models = None
        self.model_res = None
        self.window_size = 0
        self.max_layer_size = max_layer_size
        self.file_name = file_name

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
        include the training part in the set) -> calculate the imfs based on whole hsitory but return only from this
        point onwards
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
                               train_select_split=split,
                               max_layer_size=self.max_layer_size)
        model.construct_GMDH(train_x, train_y, select_x, select_y, stop_leniency=3)
        return model

    # DONE
    def train_sets(self, fitness_fns, splits: tuple[float, float, float] = (0.75 * 0.8, 0.8, 1),
                   window_size: int = 30, predict_steps: int = 10, imfs_save: str = "imfs.pickle",
                   models_save: str = "models.pickle") -> None:
        """
        Calculates the imfs for the ensambles and trains the corresponding models

        :param splits: ratio between train and selection set
        :return: None
        """
        self.window_size = window_size
        self.models = []
        imfs_and_res = {}
        #train and selection
        if os.path.exists(imfs_save):
            with open(imfs_save, 'rb') as f:
                imfs_and_res = pkl.load(f)
                imfs_train, res_train = imfs_and_res["train"]
                imfs_select, res_select = imfs_and_res["select"]
        else:
            imfs_and_res['train'] = imfs_train, res_train = self.create_median(
                *self.create_ensamble_imfs(use_split=splits[0]))

            imfs_and_res['select'] = imfs_select, res_select = self.create_median(
                *self.create_ensamble_imfs(use_split=splits[1], cut_off=floor(len(self.timeseries) * splits[0])))
            with open(imfs_save, 'wb') as f:
                pkl.dump(imfs_and_res, f)
        del imfs_and_res
        if os.path.exists(models_save):
            with open(models_save, 'rb') as f:
                self.models, self.model_res = pkl.load(f)


        # cut out only the relevant part

        for i, imf in enumerate(zip(imfs_train, imfs_select)): #imfs_test)):
            if i < len(self.models):
                print(f"{i+1} model has already been trained and saved, skipping")
                continue

            print(f"Training on {i+1} imf:")
            dloader_train = DataLoader(imf[0])
            dloader_select = DataLoader(imf[1])
            # dloader_test = DataLoader(imf[2])

            train_split = dloader_train.window_split_x_y(window_size=window_size)
            select_split = dloader_select.window_split_x_y(window_size=window_size)

            #print(f"{train_split[0].shape} , {train_split[1].shape}")
            #print(f"{select_split[0].shape} , {select_split[1].shape}")
            self.models.append(self.gmdh_train(*train_split, *select_split,
                                               fitness_fn=fitness_fns,
                                               err_function=mean_square_error))
            with open(models_save, 'wb') as f:
                pkl.dump((self.models, None), f)

        if not self.model_res:
            dloader_train = DataLoader(res_train)
            dloader_select = DataLoader(res_select)
            # dloader_test = DataLoader(res_test)
            train_split = dloader_train.window_split_x_y(window_size=window_size)
            select_split = dloader_select.window_split_x_y(window_size=window_size)
            self.model_res = self.gmdh_train(*train_split, *select_split, mean_square_error)

            with open(models_save, 'wb') as f:
                pkl.dump((self.models, self.model_res), f)
        else:
            print("Model on res already trained")

        # we need start of the test set and then move it forward by 10 each time and predicting it
        test_set_length = len(self.timeseries)
        error = []
        for i in range(test_set_length*splits[1], test_set_length, predict_steps):
            error.append(self.eval(self.timeseries, i, predict_steps, y=self.timeseries[i: i+predict_steps]))
        return error


    def eval(self, whole_ts: np.ndarray, start_index: int, no_steps_to_predict: int = 10,
             y: np.ndarray = None) -> np.ndarray | float:
        """
        Predict
        Make sure that time series is of the maximum length possible

        :param whole_ts: the whole history of the timeseries available
        :param start_index: start of slice of time series that we wish to predict
        :param no_steps_to_predict: have many steps ahead to predict
        :param y: if added it is used to calculate the mean square error make sure that y is the length of
        no_steps_to_predict
        :return: returns the predicted result or if y was given the mean square error
        """
        # TODO implement the following steps:
        # calculate imfs on the whole historical context
        self.timeseries = whole_ts[:start_index]
        # just get the last <window_size> part of the imfs and res because we will be predicting only new poitns.
        imfs, res = self.create_median(*self.create_ensamble_imfs(use_split=1, cut_off=-(self.window_size-1)))

        prediction = self.predict_based_on_imf_res(imfs, res, no_steps_to_predict)
        if y:
            return utils.mean_square_error(prediction[:len(y)], y)
        return prediction

    def predict_based_on_imf_res(self, imfs, res, no_steps_predict: int = 10):
        # predict no_steps_to_predict ahead
        for i, imf in enumerate(imfs):
            for step in range(no_steps_predict):
                X = imf[-(self.window_size - 1):]
                imf.append(self.models[i].evaluate(np.expand_dims(X, 0)))
            imfs[i] = imf
        for step in range(no_steps_predict):
            X = res[-(self.window_size - 1):]
            res.append(self.model_res.evaluate(np.expand_dims(X, 0)))
        # sum the models
        sum_all = imfs[0]
        for imf in imfs:
            sum_all = sum_all + imf
        return sum_all + res

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
        end_model = MEEMDGMDH(s, "test1.pkl")
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
        meme = MEEMDGMDH(s, 16,"MEEMD-GMDH_test2.pkl")
        meme.train_sets([(utils.poly, lambda x: x),
                         (utils.sigmoid, utils.inverse_sigmoid),
                         (utils.hyperbolic_tangent, utils.inverse_hyperbolic_tangent),
                         (utils.radial_basis, utils.inverse_radial_basis)
                         ],
                        window_size=16
                        )

