import torch
import torch.nn as nn

from torchinfo import summary

class BCGEncoderBiLSTM(nn.Module):
    def __init__(self, input_channels=1, cnn_out_channels=32, lstm_hidden=64, projection_dim=64):
        super(BCGEncoderBiLSTM, self).__init__()

        # CNN 模組：提取區域訊號特徵
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.MaxPool1d(kernel_size=5, stride=5)  # 長度從 1000 ➜ 200
        )

        # Bi-LSTM 模組：捕捉時序前後關係
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True  # 雙向 LSTM
        )

        # Projection Head：對比學習特徵映射（壓縮至 64 維）
        self.projector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        """
        x: (batch, 1, 1000)
        """
        x = self.cnn(x)              # ➜ (batch, 32, 200)
        x = x.permute(0, 2, 1)       # ➜ (batch, 200, 32) for LSTM
        _, (h_n, _) = self.lstm(x)   # h_n: (2, batch, hidden)

        # Concatenate forward + backward
        h_n = h_n.permute(1, 0, 2)   # ➜ (batch, 2, hidden)
        h_n = h_n.reshape(h_n.size(0), -1)  # ➜ (batch, 2*hidden)

        z = self.projector(h_n)      # ➜ (batch, projection_dim=64)

        return z
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    # 建立模型
    model = BCGEncoderBiLSTM(input_channels=1, cnn_out_channels=32, lstm_hidden=64, projection_dim=64)

    # 使用 torchinfo 查看模型摘要
    summary(model, input_size=(8, 1, 1000))  # (batch_size, channels, height, width)