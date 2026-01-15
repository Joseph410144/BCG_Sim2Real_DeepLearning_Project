import torch
import torch.nn as nn

class LinearBCG(nn.Module):
    """
    Input:  [B, 1, 1000]
    Output: [B, 1, 1000]
    結構：1000 -> 512 -> 1000，中間含 Dropout
    參數量 ≈ 1,025,512 (~4.1 MB, fp32)
    """
    def __init__(self, seq_len=1000, p_drop=0.2, hidden=512):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),        # [B, 1, 1000] -> [B, 1000]
            nn.Linear(seq_len, hidden, bias=True),     # 1000 -> 512
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, seq_len, bias=True),     # 512 -> 1000
            # 可選：nn.Tanh()  # 若想限制輸出範圍再打開
        )

    def forward(self, x):  # x: [B, 1, T]
        y = self.net(x)
        return y.unsqueeze(1) if y.dim()==2 else y.view(x.size(0), 1, self.seq_len)
