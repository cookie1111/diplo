# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
from emd import sift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import PolyLeastSquares
from time import process_time_ns

TEST = 1

class MEEMDGMDH:

    def __init__(self, ts):
        self.timeseries = ts

    def add_noise(self, noise_amp):
        return self.timeseries + np.random.normal(0, noise_amp, len(self.timeseries))

    def get_imfs(self, timeseries):
        s = sift.sift(timeseries, max_imfs=9)
        imfs = np.array(s)
        res = self.timeseries - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, procs = 1, noise_amp = 0.05):
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

    def gmdh(self, train_x, train_y, select_x, select_y, test_x, test_y, fitness_fn, polynomial, fitness_thresh):
        layers = []
        while True:



            if True:
                pass
        #return indexs, coefficients

if __name__ == '__main__':
    df = pd.read_csv("snp500.csv")
    #m = MEEMDGMDH(np.array(df['Close']))
    #imfs, res = m.create_ensamble_imfs()
    #imfs, res = m.create_median(imfs,res)
    #print(len(imfs),imfs[0].shape)
    p = PolyLeastSquares((0,1),[1,5,4,3,2,7],True)
    s = df['Close']
    ctr = 0
    if TEST == 1:
        # maybe good for parallel proccessing atm its useless doe
        start = process_time_ns()
        matrix = np.lib.stride_tricks.sliding_window_view(s[0:10],window_shape=7)
        c = p.calc_quadratic_matrix(matrix)
        m = process_time_ns() - start
        print(f"Speed quad matrix: {m}")
        ctr = 0
        start = process_time_ns()
        d = p.calc_quadratic(matrix)
        no_matrix = process_time_ns() - start
        print(f"Speed quad: {no_matrix}")
        print(f"matrix - no_matrix = {m-no_matrix}")
        print(f"res: {c}, {d}")
    if TEST == 2:
        matrix = np.lib.stride_tricks.sliding_window_view(s, window_shape=8)
        print(matrix[:, :7].shape, matrix[:, -1:].shape)
        #p.regression_of_function(matrix[:, :7], matrix[:, -1:])

