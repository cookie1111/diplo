from torch import nn
from utils import normalize_ts
import pandas as pd
from math import floor
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np

torch.set_printoptions(precision=10)

class SequenceDataset(Dataset):
    def __init__(self, ts, target_len, sequence_length=29):
        self.target_len = target_len
        self.sequence_length = sequence_length
        self.y = torch.tensor(ts[sequence_length:]).float()
        self.X = torch.tensor(ts[:-target_len]).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length+1

    def __getitem__(self, i):
        if i > self.X.shape[0] - self.sequence_length+1:
            raise IndexError(f"{i} is out of range for {self.__len__()}")
        #print("Dataloader bigus",self.X.shape, self.y.shape)
        if i == -1:
            x = torch.unsqueeze(self.X[-self.sequence_length:], dim=0)
        elif i < 0:
            x = torch.unsqueeze(self.X[-self.sequence_length+i+1:i+1], dim=0)
            #print(self.X[-self.sequence_length+i:i], self.y[i:])
        else:
            x = torch.unsqueeze(self.X[i:(i + self.sequence_length)], dim=0)

        return x, self.y[i]


def factory_func_for_train(input_dim, output_dim, train_dataset, test_dataset):
    def train_LSTM_models(x):#hidden_dim, num_layers, batch_size, lr_rate, epochs):
        #print("printing", x)
        res = []
        for i in range(len(x)):
            hidden_dim = floor(x[i][0])
            num_layers = floor(x[i][1])
            batch_size = floor(x[i][2])
            lr_rate = x[i][3]
            epochs = floor(x[i][4])

            model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            optimizer = Adam(model.parameters(), lr=lr_rate)

            loss_function = nn.MSELoss()
            test_model(test_loader, model, loss_function)

            for epoch in range(epochs):
                train_model(train_loader, model, loss_function, optimizer=optimizer)
                test_loss = test_model(test_loader, model, loss_function)
            res.append(test_loss)
        return np.array(res)

    return train_LSTM_models


def model_factory_func(input_dim, output_dim, train_test_dataset):
    def create_LSTM_model(x):#hidden_dim, num_layers, batch_size, lr_rate, epochs):
        hidden_dim = floor(x[0])
        num_layers = floor(x[1])
        batch_size = floor(x[2])
        lr_rate = x[3]
        epochs = floor(x[4])

        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

        train_test_loader = DataLoader(train_test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = Adam(model.parameters(), lr=lr_rate)

        loss_function = nn.MSELoss()
        test_model(test_loader, model, loss_function)

        for epoch in range(epochs):
            train_model(train_test_loader, model, loss_function, optimizer=optimizer)
        return model

    return create_LSTM_model


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        #print(x.size())

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(hn[0])

        return torch.squeeze(out)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def eval_model(data_loader, model: LSTM):
    output = None
    if
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

if __name__ == "__main__":
    TEST = 0

    df = pd.read_csv("snp500.csv")
    sig = normalize_ts(df["Close"].values, 0.8 * 0.8)
    sequence_length = 29
    target_length = 1

    train_dataset = SequenceDataset(sig[:floor(len(sig)*0.8*0.8)], target_len=target_length,
                                    sequence_length=sequence_length)
    test_dataset = SequenceDataset(sig[floor(len(sig)*0.8*0.8):floor(len(sig)*0.8)], target_len=target_length,
                                   sequence_length=sequence_length)
    val_dataset = SequenceDataset(sig[floor(len(sig)*0.8):], target_len=target_length,
                                  sequence_length=sequence_length)

    if TEST == 0:
        # Set the random seed for reproducibility
        torch.manual_seed(42)

        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

        input_dim = 29
        hidden_dim = 64
        num_layers = 2
        output_dim = 1

        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

        optimizer = Adam(model.parameters(), lr=0.0001)

        loss_function = nn.MSELoss()
        print("Untrained test\n--------")
        test_model(test_loader, model, loss_function)
        print()

        for ix_epoch in range(200):
            print(f"Epoch {ix_epoch}\n---------")
            train_model(train_loader, model, loss_function, optimizer=optimizer)
            test_model(test_loader, model, loss_function)
            print()

    if TEST == 1:
        df = pd.read_csv("snp500.csv")
        sig = normalize_ts(df["Close"].values, 0.8 * 0.8)
        print(sig[:30])
        print(SequenceDataset(sig, 1, 29)[1])