import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel

class RNNLayer(nn.Module):
    """
    Container module for a single LSTM layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, seq_len, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(RNNLayer, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = bidirectional

        self.rnn = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional= self.num_direction)

        # linear projection layer
        if self.num_direction:
            self.proj = nn.Linear(hidden_size * 2, input_size)
        else:
            self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

class LSTM_SingleTask(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(LSTM_SingleTask, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    # nn.Linear(in_features=seq_len, out_features=seq_len),
                                    nn.Dropout(0.1))  # 增加 Dropout

    def forward(self, x):
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input = output
            output = self.rnn1[i](input)
            output = output.permute(0, 2, 1)
            output = self.rnn_norm[i](output)
            output = output.permute(0, 2, 1)

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output
    
class LSTM_BCGFilter_Pre(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(LSTM_BCGFilter_Pre, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    # nn.Linear(in_features=seq_len, out_features=seq_len),
                                    nn.Dropout(0.1))  # 增加 Dropout

    def forward(self, x):
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input = output
            output = self.rnn1[i](input)
            output = output.permute(0, 2, 1)
            output = self.rnn_norm[i](output)
            output = output.permute(0, 2, 1)

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output

class LSTM_BCGFilter_Pre_Confidence(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(LSTM_BCGFilter_Pre_Confidence, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # 2. 你的新想法：CNN Confidence layer
        self.conv_layers = nn.Sequential( # 第一層: 融合 LSTM 特徵與原始訊號 
            nn.Conv1d(2, 32, kernel_size=5, padding=2), 
            nn.BatchNorm1d(32), 
            nn.ReLU(), 
            nn.Dropout(0.3), # 防止過擬合 # 第二層: 進一步提取特徵 
            nn.Conv1d(32, 16, kernel_size=3, padding=1), 
            nn.BatchNorm1d(16), nn.ReLU() ) # 2. 全連接層 (負責打分數) 
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B,16,T] -> [B,16,1]
        self.fc_layers = nn.Sequential( nn.Linear(16, 8), 
                                       nn.ReLU(), 
                                       nn.Linear(8, 1)) # 輸出一個數值 (Logits) 

        self.output = nn.Sequential(nn.PReLU(),
                                    # nn.Linear(in_features=seq_len, out_features=seq_len),
                                    nn.Dropout(0.1))  # 增加 Dropout

    def forward(self, x):
        # B, C, T
        output = x.permute(0, 2, 1)
        # B, T, C
        for i in range(len(self.rnn1)):
            input = output
            output = self.rnn1[i](input)
            output = output.permute(0, 2, 1)
            output = self.rnn_norm[i](output)
            output = output.permute(0, 2, 1)

        output = output.permute(0, 2, 1)
        output = self.output(output)

        # B, C, T
        combined_input = torch.cat((output, x), dim=1) # [Batch, Time, Feat+1]
        feature = self.conv_layers(combined_input)
        g = self.pool(feature).squeeze(-1)     # [B,16]
        confident_score = self.fc_layers(g)

        return output, confident_score

class LSTM_BCGFilter_Post(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 3):
        super(LSTM_BCGFilter_Post, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    # nn.Linear(in_features=seq_len, out_features=seq_len),
                                    nn.Dropout(0.1))  # 增加 Dropout

    def forward(self, x):
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input = output
            output = self.rnn1[i](input)
            output = output.permute(0, 2, 1)
            output = self.rnn_norm[i](output)
            output = output.permute(0, 2, 1)
            output = output + input
              

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output



if __name__ == '__main__':
    # 建立模型
    model = LSTM_BCGFilter_Pre_Confidence(seq_len=1000, input_size=1, hidden_size=128, output_size=1,
            dropout=0.2, num_layers=6, bidirectional=True, repeat_times = 1)

    # 使用 torchinfo 查看模型摘要
    summary(model, input_size=(8, 1, 1000))  # (batch_size, channels, height, width)