import numpy as np
import pandas as pd
from meemd_gmdh import MEEMDGMDH
from PyEMD import EMD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from utils import normalize_ts

class EEMD_Clustered_SVR_PSO_LSTM:

    def __init__(self, window_size: int, prediction_size: int):
        self.emd = EMD()


    # Maybe use entropy to compare
    def emd_calculation_and_clustering(self, ts: np.ndarray) -> dict[int: np.ndarray]:
        """
        Calculate and cluster the imfs + res
        :param ts: timeseries that the model will decompose
        :return: clustered timeseries lowest number represents the highest frequency
        """
        self.emd.emd(ts)
        IMFS = self.emd.get_imfs_and_residue()
        kmeans_estimator = KMeans(n_clusters=3, n_init=100).fit(IMFS[0])
        classified = kmeans_estimator.fit_predict(IMFS[0])
        print(classified)
        # najvišje frekvence so prve torej rabimo njihov razred prirediti
        legenda = {}
        class_counter = 0
        combined_imfs = {}
        for i in range(0, IMFS[0].shape[0]):
            if classified[i] not in legenda:
                legenda[classified[i]] = class_counter
                class_counter = class_counter + 1
                combined_imfs[legenda[classified[i]]] = IMFS[0][i, :]
            else:
                combined_imfs[legenda[classified[i]]] = combined_imfs[legenda[classified[i]]] + IMFS[0][i, :]
        combined_imfs[class_counter+1] = IMFS[1]
        return combined_imfs

    def svr_high_freq(self, high_freq_ts: np.ndarray):
        self.svr = SVR()


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
        sig = df["Close"].values[-1000:]
        plt.plot(range(len(sig)), sig)
        plt.show()
        model = EEMD_Clustered_SVR_PSO_LSTM(30, 10)
        clustered = model.emd_calculation_and_clustering(sig)
        print(len(clustered))
        for i in clustered:
            print(clustered[i].shape)
            plt.plot(range(len(clustered[i])),clustered[i] )
            plt.show()
    if TEST == 2:
        df = pd.read_csv("snp500.csv")
        sig = normalize_ts(df["Close"].values)
        model = EEMD_Clustered_SVR_PSO_LSTM(30, 10)


