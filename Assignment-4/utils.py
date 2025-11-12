import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, cell_type="RNN", bidirectional=False, num_classes=10):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.cell_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh', bidirectional=bidirectional)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("Unsupported cell_type: choose from 'RNN','LSTM','GRU'")

        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size) where seq_len = 28 for MNIST rows/cols
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
        if self.cell_type == "LSTM":
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)
        # Many-to-one: take the last time-step output (out[:, -1, :])
        last = out[:, -1, :]  # (batch, hidden_size * num_directions)
        logits = self.fc(last)
        return logits
