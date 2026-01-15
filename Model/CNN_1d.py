import torch
import torch.nn as nn

class BCG_CNN_Small(nn.Module):
    def __init__(self, seq_len=1000, channels=1, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1)   # output channel = 1
        )
    def forward(self, x):   # x: [B,1,T]
        return self.net(x)
