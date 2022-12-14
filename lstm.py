import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import normalize_ts
import pandas as pd
from math import floor
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

torch.set_printoptions(precision=10)

class SequenceDataset(Dataset):
    def __init__(self, ts, target_len, sequence_length=29):
        self.target_len = target_len
        self.sequence_length = sequence_length
        self.y = torch.tensor(ts[sequence_length+1:]).float()
        self.X = torch.tensor(ts[:-target_len]).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i):
        if i > self.X.shape[0] - self.sequence_length:
            raise IndexError(f"{i} is out of range for {self.__len__()}")

        x = self.X[i:(i + self.sequence_length)]

        return x, self.y[i]

def factory_func_for_train(input_dim, output_dim, X_train, y_train, X_test, y_test):
    def train_LSTM_models(hidden_dim, num_layers, batch_size, lr_rate, epochs):

        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        err = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, lr_rate)

    return train_LSTM_models


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

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(hn[0])

        return out


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
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")




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

        for ix_epoch in range(2):
            print(f"Epoch {ix_epoch}\n---------")
            train_model(train_loader, model, loss_function, optimizer=optimizer)
            test_model(test_loader, model, loss_function)
            print()

    if TEST == 1:
        df = pd.read_csv("snp500.csv")
        sig = normalize_ts(df["Close"].values, 0.8 * 0.8)
        print(sig[:30])
        print(SequenceDataset(sig, 1, 29)[1])