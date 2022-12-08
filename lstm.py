import torch
from torch import nn

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

def train(model, x_train, y_train, x_val, y_val, num_epochs, batch_size, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = criterion(y_val_pred, y_val)

        print(f"Epoch: {epoch + 1}/{num_epochs} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")



if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Define the model parameters
    input_dim = 10
    hidden_dim = 8
    num_layers = 2
    output_dim = 1

    # Generate dummy training and validation data
    x_train = torch.randn(100, 10, input_dim)
    y_train = torch.randint(0, 2, size=(100, output_dim))
    x_val = torch.randn(50, 10, input_dim)
    y_val = torch.randint(0, 2, size=(50, output_dim))

    # Create the model
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    print(x_train.type(),y_train.type(), x_val.type(), y_val.type())
    # Train the model
    train(model, x_train, y_train.type(torch.FloatTensor), x_val, y_val.type(torch.FloatTensor), num_epochs=5, batch_size=32, learning_rate=0.001)