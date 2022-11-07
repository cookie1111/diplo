# implemented based off https://downloads.hindawi.com/journals/mpe/2021/5589717.pdf
from emd import sift
import numpy as np


class MEEMDGMDH:

    def __init__(self, ts):
        self.timeseries = ts

    def add_noise(self, noise_amp):
        return self.timeseries + np.random.normal(0, noise_amp, len(self.timeseries))

    def get_imfs(self, timeseries):
        imfs = np.array(sift(timeseries, max_imfs = 12))
        res = self.timeseries - np.sum(imfs, axis=-1)
        return imfs, res

    def create_ensamble_imfs(self, M = 10):

