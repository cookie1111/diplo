import numpy as np
import pandas as pd
from meemd_gmdh import MEEMDGMDH
from PyEMD import EMD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class EEMD_Clustered_SVR_PSO_LSTM:

    def __init__(self, window_size: int, prediction_size: int):
        self.emd = EMD()

    def emd_calculation_and_clustering(self, ts: np.ndarray):
        self.emd.emd(ts)
        IMFS = self.emd.get_imfs_and_residue()
        kmeans_estimator = KMeans(n_clusters=3, n_init=100).fit(IMFS[0])
        print(IMFS[0].shape)
        for i in range(0,IMFS[0].shape[0]):
            plt.plot(list(range(0, len(IMFS[0][i, :]))), IMFS[0][i, :])
            plt.show()
        print(kmeans_estimator.fit_predict(IMFS[0]))
        return


if __name__ == "__main__":
    TEST = 1

    if TEST == 0:
        sig = np.linspace(0, 1, 200)
        sig = np.cos(11 * 2 * np.pi * sig * sig)
        plt.plot(np.linspace(0, 1, 200), sig)
        plt.show()
        model = EEMD_Clustered_SVR_PSO_LSTM(30, 10)
        plt.plot(np.linspace(0, 1, 200), model.emd_calculation_and_clustering(sig))
        plt.show()
    if TEST == 1:
        df = pd.read_csv("snp500.csv")
        sig = df["Close"].values
        plt.plot(range(len(sig)), sig)
        plt.show()
        model = EEMD_Clustered_SVR_PSO_LSTM(30, 10)
        plt.plot(range(len(sig)), model.emd_calculation_and_clustering(sig))
        plt.show()
