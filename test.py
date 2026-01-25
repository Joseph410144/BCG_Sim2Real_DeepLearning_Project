import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class BCG_Generator_Final:
    # ... (你的 get_random_bcg_params 和 get_full_bcg_waveform 保持上面的向量化版本不變) ...
    # 這裡省略重複的程式碼，直接進入核心新增功能
    
    def get_random_bcg_params(self, hr_freq_array):
        # (同上一版，支援向量化運算)
        hr_freq = np.array(hr_freq_array)
        
        # 標準參數
        std_a = np.array([0.3, -0.7, 1.2, -0.9, 0.5]) 
        std_b = np.array([0.3, 0.24, 0.4, 0.3, 0.4]) 
        std_t = np.array([-1.4, -0.6, 0.0, 0.6, 1.8]) 

        # === 向量化計算 ===
        width_scale = np.sqrt(1 / hr_freq) 
        
        if hr_freq.ndim > 0:
            width_scale = width_scale[np.newaxis, :] 
            new_b = std_b[:, np.newaxis] * width_scale 
            new_t = std_t[:, np.newaxis] * width_scale 
        else:
            new_b = std_b * width_scale
            new_t = std_t * width_scale

        # Amp Noise (這裡也可以加入隨機遊走，讓波形變化更自然)
        new_a = std_a[:, np.newaxis] * np.random.uniform(0.9, 1.1, size=(5, hr_freq.shape[0]))

        # Pos Jitter
        pos_jitter = np.random.uniform(-0.02, 0.02, size=(5, hr_freq.shape[0]))
        new_t = new_t + pos_jitter
        new_t[2] = 0.0 # Lock J wave
        
        return new_a, new_b, new_t

    def get_full_bcg_waveform(self, theta, hr_freq_array):
        # (同上一版)
        phase = np.angle(np.exp(1j * theta))
        a_para, b_para, t_para = self.get_random_bcg_params(hr_freq_array)
        
        waves_data = [
            (t_para[0], b_para[0], a_para[0]), 
            (t_para[1], b_para[1], a_para[1]), 
            (t_para[2], b_para[2], a_para[2]), 
            (t_para[3], b_para[3], a_para[3]), 
            (t_para[4], b_para[4], a_para[4]), 
        ]

        bcg_total = np.zeros_like(phase)
        for (mu, sigma, amp) in waves_data:
            d_theta = np.angle(np.exp(1j * (phase - mu)))
            bcg_total += amp * np.exp(-(d_theta**2) / (2 * sigma**2))

        return bcg_total

# === 🔥 核心：產生非呼吸性心率變異 (Non-Respiratory HRV) ===
def generate_intrinsic_hrv_noise(duration, fs, hrv_scale=0.5):
    """
    產生具有 1/f 特性 (Pink Noise) 的心率擾動
    hrv_scale: 變異強度 (例如 0.05 代表 5% 的變異)
    """
    n_samples = int(duration * fs)
    
    # 1. 產生白雜訊
    white_noise = np.random.randn(n_samples)
    
    # 2. 轉換到頻域
    X_white = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(n_samples, d=1/fs)
    
    # 3. 施加 1/f 濾波器 (Pink Noise)
    # 避免除以 0，第一點設為 0
    scale = 1.0 / (frequencies + 1e-10) 
    scale[0] = 0 
    
    # 4. 產生粉紅雜訊頻譜
    X_pink = X_white * np.sqrt(scale)
    
    # 5. 轉回時域並正規化
    pink_noise = np.fft.irfft(X_pink, n=n_samples)
    
    # 正規化到 -1 ~ 1 之間
    pink_noise = pink_noise / np.max(np.abs(pink_noise))
    
    return pink_noise * hrv_scale

def get_total_phase(duration, fs, mean_bpm=60, rsa_signal=None, hrv_scale=0.1):
    """
    整合：平均心率 + 呼吸調節 (RSA) + 非呼吸隨機 HRV
    """
    t = np.linspace(0, duration, int(duration*fs))
    mean_freq = mean_bpm / 60.0
    
    # 1. 基礎心率
    hr_freq = np.ones_like(t) * mean_freq
    
    # 2. 加入非呼吸性 HRV (Intrinsic Random Walk)
    # 這是你想要的「真正的隨機變異」
    intrinsic_noise = generate_intrinsic_hrv_noise(duration, fs, hrv_scale)
    hr_freq += (mean_freq * intrinsic_noise)
    
    # 3. 加入 RSA (呼吸調節) - 如果你原本就有呼吸訊號
    if rsa_signal is not None:
        # 假設 rsa_signal 已經正規化，這裡只是示範強度
        # 呼吸通常會讓心率頻率改變約 0.05 ~ 0.15 Hz
        rsa_modulation = rsa_signal * 0.1 # 調變強度
        hr_freq += rsa_modulation
        
    # 4. 積分得到相位
    theta = 2 * np.pi * np.cumsum(hr_freq) / fs
    
    return theta, hr_freq, t

# === 驗證與視覺化 ===
gen = BCG_Generator_Final()
duration = 10
fs = 100

# 假設這是你的呼吸訊號 (RSA)
resp_freq = 0.25 # 15 breaths/min
t_sim = np.linspace(0, duration, int(duration*fs))
rsa_signal = 0.5 * np.sin(2 * np.pi * resp_freq * t_sim)

# 產生相位與總心率 (混合了 RSA + Intrinsic HRV)
theta, hr_freq, t = get_total_phase(duration, fs, mean_bpm=60, rsa_signal=rsa_signal, hrv_scale=0.8)

# 產生波形
bcg = gen.get_full_bcg_waveform(theta, hr_freq)

# 畫圖
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, hr_freq * 60, label='Total HR')
plt.plot(t, (60 + rsa_signal*0.1*60), '--', alpha=0.5, label='RSA Component Only')
plt.title("Heart Rate Variability (RSA + Intrinsic 1/f Noise)")
plt.ylabel("BPM")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t, bcg)
plt.title("Synthetic BCG with Realistic HRV (Rate-Adaptive Width)")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()