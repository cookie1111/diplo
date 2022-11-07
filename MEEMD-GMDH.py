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
        s = sift.sift(timeseries, max_imfs=12)
        imfs = np.array(s)
        res = self.timeseries - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, procs = 1, noise_amp = 0.05):
        if procs > 1:
            pass
        else:
            noise_width = noise_amp * np.abs(np.max(self.timeseries) - np.min(self.timeseries))
            number_ensamble_memebers = len(self.timeseries)
            all_imfs = {}
            for i in range(number_ensamble_memebers):
                all_imfs[i] = self.get_imfs(self.add_noise(noise_width))
                print(all_imfs[i][0][:,0].shape)
                if i == 3:
                    break
            fig,ax = plt.subplots(len(all_imfs))
            for i in all_imfs:
                #print(all_imfs[i][0][:4,0])
                ax[i].plot(all_imfs[i][0][:,6])
            fig.show()


if __name__ == '__main__':
    df = pd.read_csv("snp500.csv")
    m = MEEMDGMDH(np.array(df['Close']))
    m.create_ensamble_imfs()
