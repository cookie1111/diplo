# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
import os.path
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
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


TEST = 13


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
        #print(f"{timeseries.shape},{upper_limit}")
        s = sift.sift(timeseries[:upper_limit], max_imfs=amount_of_imfs)
        #print(s.shape)
        imfs = np.array(s[cut_off:, :])
        res = self.timeseries[cut_off:upper_limit] - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, procs: int = 1, noise_amp: float = 0.05,
                             use_split: float | int = 0.75, cut_off: int = 0) -> tuple[dict[int, list[np.ndarray]],
                                                                                 list[np.ndarray]]:
        """
        Calculate imfs for the given timeseries, and correct set(train/select/val) for each set the same start is
        used, just the cut-off point changes

        :param procs: amount of processes used CURRENTLY NOT IMPLEMENTED
        :param noise_amp: amplitude of the added gaussian noise
        :param use_split: add the upper limit -> whole set == 1, 0.5 means the upper limit is at half of the set, can
        also pass the upper limit directly as an index(the limit itself is not included)
        :param cut_off: starting position for the imfs(for example if its a select set this mean that we don't want to
        include the training part in the set) -> calculate the imfs based on whole history but return only from this
        point onwards
        :return: imfs + res
        """
        if procs > 1:
            pass
        else:

            if type(use_split) is float:
                upper_limit = floor(len(self.timeseries)*use_split)
            elif type(use_split) is int:
                upper_limit = use_split
            #print(f"{upper_limit}")
            noise_width = noise_amp * np.abs(np.max(self.timeseries[:upper_limit]) - np.min(self.timeseries[
                                                                                            :upper_limit]))
            number_ensamble_memebers = 1000
            all_imfs = {}
            all_res = []
            for i in range(number_ensamble_memebers):
                imf, res = self.get_imfs(self.add_noise(noise_width), upper_limit, cut_off=cut_off)
                for j in range(imf.shape[1]):
                    #print(imf[:,j].shape)
                    if j in all_imfs:
                        all_imfs[j].append(imf[:, j])
                    else:
                        all_imfs[j] = []
                        all_imfs[j].append(imf[:, j])
                    all_res.append(res)
                #print(all_imfs[j][0].shape)
            return all_imfs, all_res

    def create_median(self, imfs, res):
        imfs_medians = []
        for i in imfs:
            #print(len(imfs),len(imfs[0]),len(imfs[0][0]))
            nup = np.median(np.array(imfs[i]),axis=0)
            imfs_medians.append(nup)
        nup = np.median(np.array(res), axis=0)
        res_median = [nup]
        #print(f"Shape of when calculating median {res_median[0].shape}, {imfs_medians[0].shape} ")
        # shape the same
        return imfs_medians, res_median

    def gmdh_train(self, train_x, train_y, select_x, select_y,
                   fitness_fn: tuple[tuple[Callable]] | list[tuple[Callable]] =
                   ((utils.poly, lambda x: x),
                    (utils.sigmoid, utils.inverse_sigmoid),
                    (utils.hyperbolic_tangent,utils.inverse_hyperbolic_tangent),
                    (utils.radial_basis, utils.inverse_radial_basis)),
                   err_function: Callable = mean_square_error,
                   split: float = 0.75,
                   stop_leniency: int = 1):
        # print(train_x.shape, train_y.shape)
        model = utils.GMDHSlim(transfer_functions=fitness_fn,
                               error_function=err_function,
                               train_select_split=split,
                               max_layer_size=self.max_layer_size)
        model.construct_GMDH(train_x, train_y, select_x, select_y, stop_leniency=stop_leniency)
        return model

    # DONE
    def train_sets(self, fitness_fns: list[Callable], splits: tuple[float, float, float] = (0.75 * 0.8, 0.8, 1),
                   window_size: int = 30, predict_steps: int = 10, stop_leniency: int = 1,
                   imfs_save: str = "imfs.pickle", models_save: str = "models.pickle") -> list[float]:
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
                                               err_function=mean_square_error,
                                               stop_leniency=stop_leniency))
            with open(models_save, 'wb') as f:
                pkl.dump((self.models, None), f)

        if not self.model_res:
            print(res_train, res_select)
            dloader_train = DataLoader(res_train[0])
            dloader_select = DataLoader(res_select[0])
            # dloader_test = DataLoader(res_test)
            train_split = dloader_train.window_split_x_y(window_size=window_size)
            select_split = dloader_select.window_split_x_y(window_size=window_size)
            self.model_res = self.gmdh_train(*train_split, *select_split, fitness_fn=fitness_fns,
                                             err_function=mean_square_error, stop_leniency=stop_leniency)

            with open(models_save, 'wb') as f:
                pkl.dump((self.models, self.model_res), f)
        else:
            print("Model on res already trained")

        # Turns out that i don't need to recalculate the imfs each time
        # we need start of the test set and then move it forward by 10 each time and predicting it
        test_set_length = len(self.timeseries)
        ts = self.timeseries
        plt.figure()
        imfs, res = self.create_median(*self.create_ensamble_imfs(use_split=1.0,
                                                                  cut_off=-(self.window_size-1)))
        #print(imfs[0].shape,res[0].shape)
        plt.plot(range(int(floor(test_set_length*splits[1])), test_set_length),ts[int(floor(test_set_length*splits[1])):])
        error = []
        utils.printProgressBar(0,test_set_length -int(floor(test_set_length*splits[1])), prefix="Testing")
        if predict_steps == 1:
            evaluation_all = []

        for i in range(int(floor(test_set_length*splits[1])), test_set_length, predict_steps):
            evaluation = self.eval(ts, i, predict_steps, y=ts[i: i + predict_steps],imfs=imfs, res=res )
            if predict_steps == 1:
                evaluation_all.append(evaluation)
            else:
                pass
                #plt.plot(range(i-1, i+predict_steps), evaluation[1][-(predict_steps+1):])
            error.append(evaluation[0])
            utils.printProgressBar(i+1-int(floor(test_set_length*splits[1])),
                                   test_set_length-int(floor(test_set_length*splits[1])), prefix="Testing")
        plt.show()
        plt.figure()
        plt.plot(range(len(error)), error)
        plt.show()
        return error

    def eval(self, whole_ts: np.ndarray, start_index: int, no_steps_to_predict: int = 10,
             y: np.ndarray = None, imfs: list[np.ndarray] = None,
             res: list[np.ndarray] = None) -> tuple[float, np.ndarray]:
        """
        Predict
        Make sure that time series is of the maximum length possible

        :param whole_ts: the whole history of the timeseries available
        :param start_index: start of slice of time series that we wish to predict
        :param no_steps_to_predict: have many steps ahead to predict
        :param y: if added it is used to calculate the mean square error make sure that y is the length of
        no_steps_to_predict
        :param imfs: precomputd imfs to use
        :param res: precomputed res to use
        :return: returns the predicted result or if y was given the mean square error
        """
        if imfs is None and res is None:
            # calculate imfs on the whole historical context
            self.timeseries = whole_ts[:start_index]
            #print(f"eval ts size {self.timeseries.shape}")
            # just get the last <window_size> part of the imfs and res because we will be predicting only new poitns.
            imfs, res = self.create_median(*self.create_ensamble_imfs(use_split=1.0, cut_off=-(self.window_size-1)))
        else:

            for i , imf in enumerate(imfs):
                #print("Pre imf:", imfs[i].shape)
                imfs[i] = imf[:start_index]
                #print("Post:", imfs[i].shape)
            #print("Pre res:", res[0].shape)
            res[0] = res[0][:start_index]
            #print("Post:", res[0].shape)
        # something is going wrong... with the sizes of imfs and res
        prediction = self.predict_based_on_imf_res(imfs, res, no_steps_to_predict)
        #print("Done predicting:",imfs[0].shape, res[0].shape)
        #print(prediction)
        if y is not None:
            #print("To compare with:", y)
            return utils.mean_square_error(prediction[:len(y)], y), prediction
        return 0, prediction

    def layers_composition(self):
        for no, mo in enumerate(self.models):
            print(f"Model for imf {no}:")
            mo.model_composition()
        print("Model for res:")
        self.model_res.model_composition()

    def predict_based_on_imf_res(self, imfs, res, no_steps_predict: int = 10):
        # predict no_steps_to_predict ahead
        #print(imfs, res)
        local_imfs = []
        for i, imf in enumerate(imfs):
            for step in range(no_steps_predict):
                #print(imf.shape)
                X = imf[-(self.window_size - 1):]
                #print("Imf shape:", X.shape, imf.shape)
                result = self.models[i].evaluate(np.expand_dims(X, 0))
                c = np.concatenate((imf, np.squeeze(result, axis=-1)), axis=-1)
            local_imfs.append(c)
            #print(len(c))
        res = res[0]
        for step in range(no_steps_predict):
            X = res[-(self.window_size - 1):]
            #print("Res shape:", X.shape, res.shape)
            result = self.model_res.evaluate(np.expand_dims(X, 0))
            #print(result.shape)
            try:
                res = np.concatenate((res, np.squeeze(result, axis=-1)), axis=0)
            except ValueError as e:
                print(e, res.shape, result.shape)
                return None

        # sum the models
        #print(imfs,res)
        sum_all = local_imfs[0]
        for imf in local_imfs[1:]:
            sum_all = sum_all + imf
        #print("Sum all:",sum_all)
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
        meme = MEEMDGMDH(s, 64, "MEEMD-GMDH_test64.pickle")
        meme.train_sets([(utils.poly, lambda x: x),
                         (utils.sigmoid, utils.inverse_sigmoid),
                         (utils.hyperbolic_tangent, utils.inverse_hyperbolic_tangent),
                         (utils.radial_basis, utils.inverse_radial_basis)
                         ],
                        window_size=10,
                        models_save="64model.pickle",
                        )
    if TEST == 12:
        # Comparing how different IMF is if we calculate it for longer periods and where the difference gets out of hand
        statistical_differences = {}
        comparative = [1, 7, 30, 60, 120, 180, 240, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 1800]
        #si = np.sin(0.1 * np.array(list(range(100)))) * 10
        #ts = np.random.uniform(-1, 1, size=(100,))
        #sig = si + ts
        s = utils.normalize_ts(s,0.8)
        meme = MEEMDGMDH(s, 8, "testing_imfs.pickle")
        upper_limit = floor(len(s)*0.8)
        base = meme.create_median(*meme.create_ensamble_imfs(use_split=upper_limit))
        base_dict = {"diff": 0}
        for i in range(9):
            base_dict[i] = 0
        base_dict["res"] = 0
        df = pd.DataFrame(base_dict, index=[0])
        utils.printProgressBar(0, len(comparative))
        for j, i in enumerate(comparative):
            utils.printProgressBar(j+1, len(comparative))
            # check diff between the individual IMFS
            # change split ratio to reflect the different length of the
            compared = meme.create_median(*meme.create_ensamble_imfs(use_split=upper_limit+i))
            statistical_differences[i] = {}
            statistical_differences[i]["diff"] = i
            for it, imf in enumerate(zip(base[0], compared[0])):
                #print(imf[0].shape, imf[1].shape, upper_limit)
                statistical_differences[i][it] = utils.mean_square_error(imf[0], imf[1][:upper_limit])
            #print(f"Res: {len(base[1][0])}, {len(compared[1][0])}")
            statistical_differences[i]["res"] = utils.mean_square_error(base[1][0], compared[1][0][:upper_limit])
            df_inter = pd.DataFrame(statistical_differences[i], index=[i])
            #print(df_inter)
            df = pd.concat((df, df_inter), ignore_index=True)
        print(df)
        df.to_csv("statistical_importance.csv")

    if TEST == 13:
        s = utils.normalize_ts(s, 0.8)
        meme = MEEMDGMDH(s, 8, "MEEMD-GMDH_test64.pickle")
        meme.train_sets([(utils.poly, lambda x: x),
                         (utils.sigmoid, utils.inverse_sigmoid),
                         #(utils.hyperbolic_tangent, utils.inverse_hyperbolic_tangent),
                         #(utils.radial_basis, utils.inverse_radial_basis)
                         ],
                        splits=(0.8*0.8, 0.8, 1),
                        window_size=10,
                        predict_steps=1,
                        imfs_save="imf_testing.pickle",
                        models_save="model_testing.pickle",
                        )
