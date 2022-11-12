# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
from emd import sift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    df = pd.read_csv("snp500.csv")
    m = MEEMDGMDH(np.array(df['Close']))
    imfs, res = m.create_ensamble_imfs()
    imfs, res = m.create_median(imfs,res)
    print(len(imfs),imfs[0].shape)
