import torch, math
import torch.nn as nn
import torch.nn.functional as F

class MorletCWTLoss(nn.Module):
    def __init__(self, fs=100, fmin=0.7, fmax=10.0, num_freqs=48,
                 kernel_size=401, sigma=0.3, log_compress=True, eps=1e-8):
        """
        fs: sample rate
        fmin,fmax: frequency band in Hz
        num_freqs: number of wavelets (channels)
        kernel_size: length of wavelet kernel (odd recommended)
        sigma: Gaussian envelope width (in seconds-ish via scaling below; keep ~0.2-0.5)
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建議用奇數（對稱）"
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.num_freqs = num_freqs
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.log_compress = log_compress
        self.eps = eps

        # log-spaced frequencies usually work better for physiology
        freqs = torch.logspace(math.log10(fmin), math.log10(fmax), steps=num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)

        # build wavelet bank
        t = torch.arange(-(kernel_size//2), kernel_size//2 + 1) / fs  # seconds
        t = t[None, :]  # [1, K]

        real_bank = []
        imag_bank = []

        for f0 in freqs:
            # Morlet: exp(-t^2/(2*s^2)) * exp(j*2π f0 t)
            # choose s proportional to 1/f0 so low freq has wider window
            s = (self.sigma / f0)  # seconds
            gauss = torch.exp(-0.5 * (t / (s + 1e-12))**2)  # [1,K]
            cos = torch.cos(2 * math.pi * f0 * t)
            sin = torch.sin(2 * math.pi * f0 * t)

            real = gauss * cos
            imag = gauss * sin

            # normalize energy so each filter has comparable gain
            real = real / (real.pow(2).sum().sqrt() + eps)
            imag = imag / (imag.pow(2).sum().sqrt() + eps)

            real_bank.append(real)
            imag_bank.append(imag)

        # [F, 1, K]
        real_filters = torch.stack(real_bank, dim=0)
        imag_filters = torch.stack(imag_bank, dim=0)

        self.register_buffer("real_filters", real_filters, persistent=False)
        self.register_buffer("imag_filters", imag_filters, persistent=False)

    def _scalogram(self, x):
        # x: [B,1,T]
        Re = F.conv1d(x, self.real_filters, padding=self.kernel_size//2)
        Im = F.conv1d(x, self.imag_filters, padding=self.kernel_size//2)
        mag = torch.sqrt(Re*Re + Im*Im + self.eps)  # [B, F, T]
        if self.log_compress:
            mag = torch.log1p(mag)
        return mag

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        S_pred = self._scalogram(pred)
        S_tgt  = self._scalogram(target)

        return F.l1_loss(S_pred, S_tgt)

class MultiResolutionSTFTLoss(nn.Module):
    """
    BCG 重建的黃金標準 Loss:
    同時在多個 FFT 解析度下計算 Spectral Convergence 和 Log Magnitude Loss。
    這能確保模型學出的 I-J-K 波形既「準確」又「銳利」。
    """
    def __init__(self, 
                 fft_sizes=[64, 256, 512], 
                 hop_sizes=[16, 64, 128], 
                 win_lengths=[32, 128, 256],
                 window="hann_window"):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def stft(self, x, fft_size, hop_size, win_length):
        # 建立視窗函數 (Hann window)
        window = getattr(torch, self.window)(win_length).to(x.device)
        # 進行 STFT
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        # 取 Magnitude (振幅譜)
        x_mag = torch.abs(x_stft)
        
        # 防止 log(0)
        return x_mag + 1e-7

    def forward(self, y_pred, y_true):
        """
        y_pred: 模型重建的波形 (Batch, Time)
        y_true: Ground Truth 波形 (Batch, Time)
        """
        # 如果輸入是 (Batch, 1, Time)，先 squeeze 成 (Batch, Time)
        if y_pred.dim() == 3: y_pred = y_pred.squeeze(1)
        if y_true.dim() == 3: y_true = y_true.squeeze(1)

        sc_loss = 0.0
        mag_loss = 0.0

        # 對每一組解析度 (Resolution) 算 Loss
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mag_pred = self.stft(y_pred, fs, hs, wl)
            mag_true = self.stft(y_true, fs, hs, wl)

            # 1. Spectral Convergence Loss (頻譜收斂損失)
            # 讓整體頻譜能量分布正確
            sc_loss += torch.norm(mag_true - mag_pred, p="fro") / torch.norm(mag_true, p="fro")

            # 2. Log Magnitude Loss (對數振幅損失)
            # 讓細節 (微弱的高頻諧波) 也能被學到
            mag_loss += F.l1_loss(torch.log(mag_true), torch.log(mag_pred))

        # 平均化
        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)

        return sc_loss #+ mag_loss
    
class EnvelopeCorrelationLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("_cached_h", torch.empty(0))
        self._cached_N = None

    def _get_h(self, N: int, device):
        # Cache Hilbert filter (real-valued, length N)
        if self._cached_N != N or self._cached_h.numel() != N or self._cached_h.device != device:
            h = torch.zeros(N, device=device)
            h[0] = 1
            h[1:(N + 1) // 2] = 2
            if N % 2 == 0:
                h[N // 2] = 1
            self._cached_h = h
            self._cached_N = N
        return self._cached_h

    def get_envelope(self, x_2d):
        # x_2d: [B, T]
        N = x_2d.shape[-1]
        Xf = torch.fft.fft(x_2d)                       # [B, T] complex
        h = self._get_h(N, x_2d.device)[None, :]        # [1, T] real
        analytic = torch.fft.ifft(Xf * h)               # [B, T] complex
        envelope = torch.abs(analytic)                  # [B, T] real
        return envelope

    def forward(self, pred_bcg, ref_ecg):
        # 支援 [B, 1, T] 或 [B, T]
        if pred_bcg.dim() == 3:
            pred_bcg = pred_bcg.squeeze(1)  # [B, T]
        if ref_ecg.dim() == 3:
            ref_ecg = ref_ecg.squeeze(1)    # [B, T]

        # 1) envelope
        env_pred = self.get_envelope(pred_bcg)
        env_ref  = self.get_envelope(ref_ecg)

        # 2) normalize per-sample
        env_pred = (env_pred - env_pred.mean(dim=-1, keepdim=True)) / (env_pred.std(dim=-1, keepdim=True) + self.eps)
        env_ref  = (env_ref  - env_ref.mean(dim=-1, keepdim=True)) / (env_ref.std(dim=-1, keepdim=True) + self.eps)

        # 3) cosine similarity (Pearson-like)
        cosine_sim = F.cosine_similarity(env_pred, env_ref, dim=-1)  # [B]
        loss = 1.0 - cosine_sim.mean()
        return loss