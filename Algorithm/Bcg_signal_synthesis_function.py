import torch
import os, random
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
# from Filters import BandPassFilter, _draw_signal_fft_spectrum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BedBCGGenerator:
    def __init__(self, fs=100):
        self.fs = fs

    def get_random_bcg_params(self, hr_freq): 
        """ 自動生成一組隨機變異的 McSharry 參數。 策略：以一組「標準參數」為基底，進行隨機縮放與微擾。 """ 

        std_a = np.array([0.3, -0.7, 1.2, -0.9, 0.5]) 
        std_b = np.array([0.3, 0.24, 0.4, 0.3, 0.4]) 
        std_t = np.array([-1.4, -0.6, 0.0, 0.6, 1.8]) 

        width_scale = np.sqrt(1/hr_freq)
        new_b = std_b * width_scale 
        new_t = std_t * width_scale 
        amp_noise = np.random.uniform(0.8, 1.2, size=5) 
        new_a = std_a #* amp_noise 
        pos_jitter = np.random.uniform(-0.01, 0.01, size=5) 
        new_t = new_t + pos_jitter 
        new_t[2] = 0.0 
        
        return new_a, new_b, new_t

    def get_full_bcg_waveform(self, theta, hr_freq):
        """McSharry Model (保持不變)"""
        phase = np.angle(np.exp(1j * theta))
        a_para, b_para, t_para = self.get_random_bcg_params(hr_freq)
        waves = {
            # Name: (Center, Width, Amplitude)
            # 這裡的 Width (Sigma) 全部加大了
            'H': (t_para[0], b_para[0], a_para[0]),  
            'I': (t_para[1], b_para[1], a_para[1]), 
            'J': (t_para[2], b_para[2], a_para[2]),   # 主波從 0.06 -> 0.12
            'K': (t_para[3], b_para[3], a_para[3]), # K波變胖
            'L': (t_para[4], b_para[4], a_para[4]),   # 舒張期波更平緩
        }

        bcg_total = np.zeros_like(phase)
        for name, (mu, sigma, amp) in waves.items():
            d_theta = np.angle(np.exp(1j * (phase - mu)))
            bcg_total += amp * np.exp(-(d_theta**2) / (2 * sigma**2))

        return bcg_total

    def generate(self, duration=10):
        t = np.linspace(0, duration, int(duration * self.fs))
        # add great sample (85%) and bad sample (15%)
        # 1. 呼吸調變 (Resp + Harmonic) - 保持上次的大振幅與相位差
        rr_freq = self.beta_in_range(low=0.15, high=0.35, a=3, b=3, size=None)
        resp_fundamental = np.cos(2 * np.pi * rr_freq * t)
        
        # 呼吸諧波
        harmonic_amp = np.random.uniform(0.1, 0.3)
        harmonic_phase = np.random.uniform(0, 2 * np.pi) 
        resp_harmonic = harmonic_amp * np.cos(4 * np.pi * rr_freq * t + harmonic_phase)
        
        resp_signal = resp_fundamental + resp_harmonic
        resp_signal = resp_signal / np.max(np.abs(resp_signal))
        
        # 2. FM 調變 (RSA)
        hr_mean = self.beta_in_range(low=0.8, high=2, a=3, b=3, size=None)
        rsa_intensity = self.beta_in_range(low=0.05, high=0.3, a=3, b=3, size=None)
        f_inst = hr_mean + rsa_intensity * resp_signal
        theta = 2 * np.pi * np.cumsum(f_inst) / self.fs
        
        # 3. 形態生成
        bcg_pure = self.get_full_bcg_waveform(theta, hr_mean)
        # bcg_pure = self.apply_mattress_transfer_function(bcg_pure) * 3
        
        # 4. AM 調變
        am_depth = self.beta_in_range(low=0.05, high=0.3, a=3, b=3, size=None) 
        shift = int(np.random.uniform(0, self.fs/rr_freq))
        resp_am = np.roll(resp_signal, shift)
        am_envelope = 1 + am_depth * resp_am
        target_signal = bcg_pure * am_envelope
        
        # ==========================================
        # 修改重點：雜訊全面升級
        # ==========================================
        
        # A. 呼吸基線 (維持 5~20 倍)
        baseline_amp = np.random.uniform(1) 
        baseline = baseline_amp * resp_signal
        
        # B. 隨機體動 (維持)
        if np.random.rand() > 0.5:
            move_center = np.random.uniform(0, duration)
            move_width = np.random.uniform(0.5, 1.5)
            move_amp = np.random.uniform(5.0, 20.0) 
            movement = move_amp * np.exp(-((t - move_center)**2) / (2 * move_width**2))
        else:
            movement = 0
            
        # C. 感測器底噪 (High Frequency Noise) - 大幅增強！
        # 原本是 0.3~0.5，現在提升到 0.8 ~ 1.5
        # BCG J波高度約 1.0，這代表雜訊已經把心跳淹沒了 (SNR < 0)
        noise_level = np.random.uniform(0.3, 0.5)
        white_noise = noise_level*np.random.normal(0.1, 0.5, size=len(t))
        
        # D. [新增] 環境震動干擾 (Vibration Noise)
        # 模擬 40Hz 的環境震動 (接近 Nyquist 頻率的高頻干擾)
        # 這會讓訊號看起來更「毛」，更難處理
        vib_freq = np.random.uniform(35, 45) 
        vib_amp = np.random.uniform(0.1, 0.5)
        vib_noise = vib_amp * np.sin(2 * np.pi * vib_freq * t + np.random.rand()*2*np.pi)
        
        # 總雜訊
        total_high_freq_noise = white_noise + vib_noise
        
        input_signal = target_signal + baseline + total_high_freq_noise #+ movement
        input_signal = self.apply_mattress_transfer_function(input_signal)
            
        return input_signal.astype(np.float32), bcg_pure.astype(np.float32)
    
    def beta_in_range(self, low, high, a=3, b=3, size=None) -> float:
        """保留你的 Beta 分佈生成器，這很好用"""
        u = np.random.beta(a, b, size=size)
        return low + (high-low)*u
    
    def apply_mattress_transfer_function(self, input_signal):
        """
        模擬真實物理世界的衰減：
        人體組織 + 床墊泡棉 = 強力的低通濾波器
        """
        # 真實 BCG 的有效能量大多集中在 5Hz 以下
        # 設定截止頻率 (Cutoff) 為 6 Hz (這是一個很典型的床墊參數)
        cutoff_freq = self.beta_in_range(low=10, high=20, a=3, b=3, size=None)
        order = 2  # 二階濾波器 (模擬機械阻尼)
        
        sos = signal.butter(order, cutoff_freq, btype='low', fs=self.fs, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, input_signal)
        
        return filtered_signal

class BedBCGGenerator_V3:
    def __init__(self, fs=100):
        self.fs = fs

    def get_full_bcg_waveform(self, theta):
        """McSharry Model (保持不變)"""
        phase = np.angle(np.exp(1j * theta))
        waves = {
            # Name: (Center, Width, Amplitude)
            # 這裡的 Width (Sigma) 全部加大了
            'H': (-0.40, 0.10, np.random.uniform(0.05, 0.1)),  
            'I': (-0.20, 0.09, np.random.uniform(-0.2, -0.4)), 
            'J': ( 0.00, 0.12, np.random.uniform(0.9, 1.1)),   # 主波從 0.06 -> 0.12
            'K': ( 0.20, 0.12, np.random.uniform(-0.5, -0.8)), # K波變胖
            'L': ( 0.50, 0.15, np.random.uniform(0.2, 0.3)),   # 舒張期波更平緩
            'M': ( 0.75, 0.15, np.random.uniform(-0.1, -0.2)),
            'N': ( 0.95, 0.15, np.random.uniform(0.05, 0.1)),
        }
        bcg_total = np.zeros_like(phase)
        for name, (mu, sigma, amp) in waves.items():
            d_theta = np.angle(np.exp(1j * (phase - mu)))
            bcg_total += amp * np.exp(-(d_theta**2) / (2 * sigma**2))
        return bcg_total

    def generate(self, duration=10):
        t = np.linspace(0, duration, int(duration * self.fs))
        # add great sample (85%) and bad sample (15%)
        rand_case = np.random.rand()
        if rand_case <= 0.8:
            # 1. 呼吸調變 (Resp + Harmonic) - 保持上次的大振幅與相位差
            rr_freq = self.beta_in_range(low=0.15, high=0.35, a=3, b=3, size=None) #np.random.uniform(0.15, 0.35) 
            resp_fundamental = np.cos(2 * np.pi * rr_freq * t)
            
            # 呼吸諧波
            harmonic_amp = np.random.uniform(0.1, 0.2)
            harmonic_phase = np.random.uniform(0, 2 * np.pi) 
            resp_harmonic = harmonic_amp * np.cos(4 * np.pi * rr_freq * t + harmonic_phase)
            
            resp_signal = resp_fundamental + resp_harmonic
            resp_signal = resp_signal / np.max(np.abs(resp_signal))
            
            # 2. FM 調變 (RSA)
            hr_mean = self.beta_in_range(low=0.8, high=2, a=3, b=3, size=None) #np.random.uniform(0.8, 2) 
            rsa_intensity = self.beta_in_range(low=0.05, high=0.15, a=3, b=3, size=None) #np.random.uniform(0.05, 0.15)
            f_inst = hr_mean + rsa_intensity * resp_signal
            theta = 2 * np.pi * np.cumsum(f_inst) / self.fs
            
            # 3. 形態生成
            bcg_pure = self.get_full_bcg_waveform(theta)
            # bcg_pure = self.apply_mattress_transfer_function(bcg_pure) * 3
            
            # 4. AM 調變
            am_depth = self.beta_in_range(low=0.05, high=0.3, a=3, b=3, size=None) #np.random.uniform(0.05, 0.3)
            shift = int(np.random.uniform(0, self.fs/rr_freq))
            resp_am = np.roll(resp_signal, shift)
            am_envelope = 1 + am_depth * resp_am
            target_signal = bcg_pure * am_envelope
            
            # ==========================================
            # 修改重點：雜訊全面升級
            # ==========================================
            
            # A. 呼吸基線 (維持 5~20 倍)
            baseline_amp = np.random.uniform(1) 
            baseline = baseline_amp * resp_signal
            
            # B. 隨機體動 (維持)
            if np.random.rand() > 0.5:
                move_center = np.random.uniform(0, duration)
                move_width = np.random.uniform(0.5, 1.5)
                move_amp = np.random.uniform(5.0, 20.0) 
                movement = move_amp * np.exp(-((t - move_center)**2) / (2 * move_width**2))
            else:
                movement = 0
                
            # C. 感測器底噪 (High Frequency Noise) - 大幅增強！
            # 原本是 0.3~0.5，現在提升到 0.8 ~ 1.5
            # BCG J波高度約 1.0，這代表雜訊已經把心跳淹沒了 (SNR < 0)
            # noise_level = np.random.uniform(0.3, 0.5)
            white_noise = np.random.normal(0, 0.2, size=len(t))
            
            # D. [新增] 環境震動干擾 (Vibration Noise)
            # 模擬 40Hz 的環境震動 (接近 Nyquist 頻率的高頻干擾)
            # 這會讓訊號看起來更「毛」，更難處理
            vib_freq = np.random.uniform(35, 45) 
            vib_amp = np.random.uniform(0.2, 0.5)
            vib_noise = vib_amp * np.sin(2 * np.pi * vib_freq * t + np.random.rand()*2*np.pi)
            
            # 總雜訊
            total_high_freq_noise = white_noise + vib_noise
            
            input_signal = target_signal + baseline + total_high_freq_noise #+ movement

            # clean data
            label_conf = 1
        
        elif rand_case<=0.9:
            input_signal = np.random.normal(0, 0.02, len(t))
            bcg_pure = np.zeros_like(t)
            
            # all noise data
            label_conf = 0
        
        else:
            # 1. 呼吸調變 (Resp + Harmonic) - 保持上次的大振幅與相位差
            rr_freq = self.beta_in_range(low=0.15, high=0.35, a=3, b=3, size=None) #np.random.uniform(0.15, 0.35) 
            resp_fundamental = np.cos(2 * np.pi * rr_freq * t)
            
            # 呼吸諧波
            harmonic_amp = np.random.uniform(0.1, 0.2)
            harmonic_phase = np.random.uniform(0, 2 * np.pi) 
            resp_harmonic = harmonic_amp * np.cos(4 * np.pi * rr_freq * t + harmonic_phase)
            
            resp_signal = resp_fundamental + resp_harmonic
            resp_signal = resp_signal / np.max(np.abs(resp_signal))
            
            # 2. FM 調變 (RSA)
            hr_mean = self.beta_in_range(low=0.8, high=2, a=3, b=3, size=None) #np.random.uniform(0.8, 2) 
            # change RSA intensity to create noisy data
            rsa_intensity = self.beta_in_range(low=2, high=5, a=3, b=3, size=None) #np.random.uniform(0.05, 0.15)
            f_inst = hr_mean + rsa_intensity * resp_signal
            theta = 2 * np.pi * np.cumsum(f_inst) / self.fs
            
            # 3. 形態生成
            bcg_pure = self.get_full_bcg_waveform(theta)
            # bcg_pure = self.apply_mattress_transfer_function(bcg_pure) * 3
            
            # 4. AM 調變
            # change AM modulation intensity to create noisy data
            am_depth = self.beta_in_range(low=2, high=5, a=3, b=3, size=None) #np.random.uniform(0.05, 0.3)
            shift = int(np.random.uniform(0, self.fs/rr_freq))
            resp_am = np.roll(resp_signal, shift)
            am_envelope = 1 + am_depth * resp_am
            target_signal = bcg_pure * am_envelope
            
            # A. 呼吸基線 (維持 5~20 倍)
            baseline_amp = np.random.uniform(10, 30) 
            baseline = baseline_amp * resp_signal

            # B. 隨機體動 (維持)
            if np.random.rand() > 0.5:
                move_center = np.random.uniform(0, duration)
                move_width = np.random.uniform(0.5, 1.5)
                move_amp = np.random.uniform(20.0, 50.0) 
                movement = move_amp * np.exp(-((t - move_center)**2) / (2 * move_width**2))
            else:
                movement = 0
                
            # C. 感測器底噪 (High Frequency Noise) - 大幅增強！
            # 原本是 0.3~0.5，現在提升到 0.8 ~ 1.5
            # BCG J波高度約 1.0，這代表雜訊已經把心跳淹沒了 (SNR < 0)
            # noise_level = np.random.uniform(0.3, 0.5)
            white_noise = np.random.normal(0, 0.2, size=len(t))
            
            # D. [新增] 環境震動干擾 (Vibration Noise)
            # 模擬 40Hz 的環境震動 (接近 Nyquist 頻率的高頻干擾)
            # 這會讓訊號看起來更「毛」，更難處理
            vib_freq = np.random.uniform(35, 45) 
            vib_amp = np.random.uniform(10, 20)
            vib_noise = vib_amp * np.sin(2 * np.pi * vib_freq * t + np.random.rand()*2*np.pi)
            
            # 總雜訊
            total_high_freq_noise = white_noise + vib_noise
            
            input_signal = target_signal + baseline + total_high_freq_noise + movement

            # clean data
            label_conf = 0

            
        return input_signal.astype(np.float32), bcg_pure.astype(np.float32), label_conf
    
    def beta_in_range(self, low, high, a=3, b=3, size=None) -> float:
        """保留你的 Beta 分佈生成器，這很好用"""
        u = np.random.beta(a, b, size=size)
        return low + (high-low)*u
    
    def apply_mattress_transfer_function(self, input_signal):
        """
        模擬真實物理世界的衰減：
        人體組織 + 床墊泡棉 = 強力的低通濾波器
        """
        # 真實 BCG 的有效能量大多集中在 5Hz 以下
        # 設定截止頻率 (Cutoff) 為 6 Hz (這是一個很典型的床墊參數)
        cutoff_freq = 6.0 
        order = 2  # 二階濾波器 (模擬機械阻尼)
        
        sos = signal.butter(order, cutoff_freq, btype='low', fs=self.fs, output='sos')
        filtered_signal = signal.sosfiltfilt(sos, input_signal)
        
        return filtered_signal

def beta_in_range(low, high, a=3, b=3, size=None) -> float:
    """保留你的 Beta 分佈生成器，這很好用"""
    u = np.random.beta(a, b, size=size)
    return low + (high-low)*u

def synthesize_bcg_resp_signal(t: np.ndarray) -> dict:
    """
    產生呼吸訊號，回傳 Raw (給基線用) 與 Norm (給調變用)
    """
    # 1. 設定頻率
    resp_freq = beta_in_range(0.15, 0.4) # 9~24 RPM
    
    # 2. 設定呼吸相位 (t 是向量)
    # 加入隨機起始相位
    phi_r = random.uniform(0, 2*np.pi)
    theta_r = 2 * np.pi * resp_freq * t + phi_r
    
    # 3. 合成波形 (加入二次諧波與相位差，模擬吸/呼不對稱)
    amp_major = beta_in_range(2, 5) # 基線漂移強度
    amp_harmonic = random.uniform(0.1, 0.3) * amp_major
    phi_diff = random.uniform(0, 2*np.pi) # 二次諧波的相位差
    
    resp_signal_raw = amp_major * np.cos(theta_r) + \
                      amp_harmonic * np.cos(2*theta_r + phi_diff)
    
    # 4. 產生正規化版本 (範圍約 -1~1)，專門給 AM 調變用
    # 避免因為 resp_signal_raw 振幅太大導致調變失控
    resp_signal_norm = resp_signal_raw / np.max(np.abs(resp_signal_raw))
    
    return {
        "raw": resp_signal_raw,   # 用於加法 (基線漂移)
        "norm": resp_signal_norm, # 用於乘法 (AM 調變)
        "theta": theta_r,         # 用於計算 RSA (FM 調變)
        "freq": resp_freq
    }

def synthesize_bcg_heart_signal(t: np.ndarray, resp_info: dict) -> np.ndarray:
    """
    產生心跳訊號，關鍵在於引入 RSA 頻率調變
    Input:
        resp_info: 從呼吸函數傳來的資訊，用於計算 RSA
    """
    # 1. 設定平均心率
    heart_freq_mean = beta_in_range(0.8, 2.5) # 48~90 BPM
    
    # 2. RSA (頻率調變 FM) 核心邏輯
    # 讓心跳相位受呼吸相位影響 -> 產生忽快忽慢的效果
    rsa_strength = beta_in_range(0.05, 0.15) # RSA 強度 (頻率偏移量)
    phi_rsa = random.uniform(0, np.pi/2)     # RSA 相位差
    phi_h_start = random.uniform(0, 2*np.pi) # 心跳起始相位
    
    # 積分公式實作: (delta_f / f_r) * sin(theta_r)
    modulation_index = rsa_strength / resp_info['freq']
    
    # 總相位 = 線性增長 + 呼吸週期性擾動
    theta_h = 2 * np.pi * heart_freq_mean * t + \
              modulation_index * np.sin(resp_info['theta'] + phi_rsa) + \
              phi_h_start

    # 3. 合成波形 (多諧波疊加)
    # 建議疊加到 5~8 倍頻，讓波形更銳利
    heart_amp_base = random.uniform(1, 2)
    heart_signal = np.zeros_like(t)
    
    num_harmonics = 6 # 疊加 6 階諧波
    for n in range(1, num_harmonics + 1):
        # 振幅衰減 (1/n)
        amp_n = (heart_amp_base / (n**1.2)) * random.uniform(0.8, 1.2)
        # 隨機相位 (關鍵！讓波形形狀每次都不同)
        phi_n = random.uniform(0, 2*np.pi)
        
        # 使用 theta_h (含 RSA)，這樣所有諧波都會一起忽快忽慢
        heart_signal += amp_n * np.cos(n * theta_h + phi_n)
        
    # 正規化心跳振幅 (讓 Deep Learning 比較好學)
    heart_signal = heart_signal / np.std(heart_signal)
        
    return heart_signal

def _draw_fft_spectrum(target_bcg, fs=100, band=10):
    target_bcg = torch.tensor(target_bcg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # pred,target: [B,C,T]
    original_fft = torch.abs(torch.fft.rfft(torch.abs(target_bcg), dim=-1))  # [B,C,F]
    freqs = torch.fft.rfftfreq(target_bcg.shape[-1], d=1.0/fs).to(target_bcg.device)
    band_mask = freqs <= band
    band_mask = band_mask
    
    original_fft = original_fft[:, :, band_mask]
    freqs = freqs[band_mask]
    
    plt.plot(freqs.squeeze().cpu().detach().numpy()[1:], original_fft[0][0].squeeze().cpu().detach().numpy()[1:], color='blue', label='original BCG spectrum')
    plt.xlabel('Frequency')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    generator = BedBCGGenerator()
    for i in range(10):
        noisy_bcg, clean_bcg = generator.generate(duration=20)
        plt.subplot(211)
        plt.plot(noisy_bcg)
        plt.xticks([])
        plt.ylabel('Amplitude')
        plt.title('Synthetic BCG')
        # plt.subplot(212)
        # plt.plot(BandPassFilter(signal_=noisy_bcg, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500))
        plt.subplot(212)
        plt.plot(clean_bcg)
        plt.ylabel('Amplitude')
        plt.title('Pure Heart BCG')
        plt.xlabel('time(points)')
        plt.show()