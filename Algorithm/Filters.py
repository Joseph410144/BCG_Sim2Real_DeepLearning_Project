import torch
import numpy as np

from scipy import signal

def BandPassFilter(signal_, lowcut, highcut, step, fs, padlen=None):
    """
    signal_ : 要濾波的訊號(1D numpy array)
    lowcut  : 帶通濾波器下限 Hz
    highcut : 帶通濾波器上限 Hz
    step    : 濾波器階數
    fs      : 取樣率
    padlen  : padding 長度(預設自動為信號長度的 1/2)
    """
    signal_ = np.asarray(signal_)
    if signal_.ndim != 1 or len(signal_) < 2:
        raise ValueError("signal_ must be a one-dimensional signal with at least two samples")
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < fs/2")
    # 設計濾波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(step, [low, high], btype='bandpass')

    # 計算 padding 長度
    if padlen is None:
        padlen = min(len(signal_) // 2, 3 * max(len(b), len(a)))  # 安全建議值
    padlen = min(int(padlen), len(signal_) - 1)
    if padlen < 1:
        raise ValueError("padlen must be at least 1")

    # 進行對稱 padding（mirror padding）
    pre = signal_[padlen-1::-1]   # 反轉前 padlen 點
    post = signal_[-1:-padlen-1:-1]  # 反轉後 padlen 點
    padded_signal = np.concatenate([pre, signal_, post])

    # 濾波
    filtered_padded = signal.filtfilt(b, a, padded_signal)

    # 去除 padding，保留原始長度
    signal_filtered = filtered_padded[padlen:-padlen]

    return signal_filtered

def HighPassFilter(signal_, cutoff, step, fs, padlen=None):
    """
    signal_ : 要濾波的訊號 (1D numpy array)
    cutoff  : 高通濾波器下限 Hz（即通過頻率下界）
    step    : 濾波器階數
    fs      : 取樣率
    padlen  : padding 長度（預設自動為信號長度的 1/2）
    """
    signal_ = np.asarray(signal_)
    if signal_.ndim != 1 or len(signal_) < 2:
        raise ValueError("signal_ must be a one-dimensional signal with at least two samples")
    if not 0 < cutoff < fs / 2:
        raise ValueError("cutoff must satisfy 0 < cutoff < fs/2")
    # 設計濾波器
    nyquist = 0.5 * fs
    highpass = cutoff / nyquist
    b, a = signal.butter(step, highpass, btype='highpass')

    # 計算 padding 長度
    if padlen is None:
        padlen = min(len(signal_) // 2, 3 * max(len(b), len(a)))  # 安全建議值
    padlen = min(int(padlen), len(signal_) - 1)
    if padlen < 1:
        raise ValueError("padlen must be at least 1")

    # 進行對稱 padding（mirror padding）
    pre = signal_[padlen-1::-1]           # 反轉前 padlen 點
    post = signal_[-1:-padlen-1:-1]       # 反轉後 padlen 點
    padded_signal = np.concatenate([pre, signal_, post])

    # 濾波
    filtered_padded = signal.filtfilt(b, a, padded_signal)

    # 去除 padding，保留原始長度
    signal_filtered = filtered_padded[padlen:-padlen]

    return signal_filtered
