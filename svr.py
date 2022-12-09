# Import cupy and the necessary sklearn functions
# import cupy as cp
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import DataLoader
import numpy as np

df = pd.read_csv("snp500.csv")
dl = DataLoader(df["Close"])


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(*dl.window_split_x_y(10, 1), test_size=0.2)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
# Convert the training and test sets to cupy arrays
#X_train_gpu = cp.array(X_train)
#y_train_gpu = cp.array(y_train)
#X_test_gpu = cp.array(X_test)
#y_test_gpu = cp.array(y_test)

# Create the SVR model
model = SVR(kernel='rbf', gamma='scale')

# Train the model on the GPU
model.fit(X_train, y_train)

# Use the model to make predictions on new data
y_pred = model.predict(X_test)
