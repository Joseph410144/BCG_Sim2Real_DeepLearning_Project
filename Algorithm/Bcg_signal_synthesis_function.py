import torch
import os, random
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.interpolate import interp1d
# from Filters import BandPassFilter, _draw_signal_fft_spectrum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BedBCGGenerator:
    def __init__(self, fs=100):
        self.fs = fs

    def generate_intrinsic_hrv_noise(self, duration, fs, hrv_scale=0.5):
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

    def get_total_phase(self, duration, fs, mean_bpm=60, rsa_signal=None, hrv_scale=0.1):
            """
            整合：平均心率 + 呼吸調節 (RSA) + 非呼吸隨機 HRV
            """
            t = np.linspace(0, duration, int(duration*fs))
            mean_freq = mean_bpm / 60.0
            
            # 1. 基礎心率
            hr_freq = np.ones_like(t) * mean_freq
            
            # 2. 加入非呼吸性 HRV (Intrinsic Random Walk)
            # 這是你想要的「真正的隨機變異」
            intrinsic_noise = self.generate_intrinsic_hrv_noise(duration, fs, hrv_scale)
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

    def generate_arrhythmia_profile(self, duration, mean_bpm=60, anomaly_prob=0.15):
        """
        產生帶有「病理性特徵」的心率曲線。
        duration: 秒數
        mean_bpm: 平均心率
        anomaly_prob: 出現心律不整事件的機率 (0.15 代表 15% 的心跳會異常)
        """
        # 1. 計算理想狀態下，大概會有幾次心跳
        avg_rr = 60.0 / mean_bpm
        num_beats = int(duration / avg_rr * 1.2) # 多算一點當緩衝
        
        # 2. 生成基礎的 RR Intervals (包含正常的微幅變異)
        # 正常人的心跳還是會有一點點抖動 (Normal sinus rhythm)
        rr_intervals = np.random.normal(loc=avg_rr, scale=0.05*avg_rr, size=num_beats)
        
        # 3. === 注入病理性事件 (Pathological Events) ===
        final_rr = []
        i = 0
        while i < len(rr_intervals):
            # 擲骰子決定這一次心跳正不正常
            is_anomaly = np.random.rand() < anomaly_prob
            
            if not is_anomaly:
                # 沒事，就把正常的 RR 加進去
                final_rr.append(rr_intervals[i])
                i += 1
            else:
                # === 發生心律不整！隨機選一種狀況 ===
                event_type = np.random.choice(['PVC', 'Missed', 'Delay'])
                
                if event_type == 'PVC': 
                    # --- 過早搏動 (Premature Ventricular Contraction) ---
                    # 特徵：這一跳很快來 (0.6x)，下一跳會補償性休息 (1.4x)
                    premature_rr = rr_intervals[i] * 0.6 
                    compensatory_pause = rr_intervals[i] * 1.4
                    final_rr.append(premature_rr)
                    final_rr.append(compensatory_pause)
                    i += 1 # 消耗掉這一次原本的額度
                    
                elif event_type == 'Missed':
                    # --- 漏拍 / 傳導阻滯 (Block) ---
                    # 特徵：直接把兩次 RR 合併，變成超長的一次間隔 (2.0x)
                    if i + 1 < len(rr_intervals):
                        missed_rr = rr_intervals[i] + rr_intervals[i+1]
                        final_rr.append(missed_rr)
                        i += 2 # 一次消耗掉兩次心跳
                    else:
                        final_rr.append(rr_intervals[i])
                        i += 1
                        
                elif event_type == 'Delay':
                    # --- 隨機延遲 (Random Delay) ---
                    # 特徵：單純慢了半拍
                    delayed_rr = rr_intervals[i] * np.random.uniform(1.2, 1.5)
                    final_rr.append(delayed_rr)
                    i += 1

        # 4. 將 RR Intervals 轉換為時間軸上的瞬時頻率 (Instantaneous Frequency)
        # 因為 McSharry 需要的是「每個時間點的頻率」來積分相位
        
        # 計算每個心跳發生的絕對時間點 (Beat Times)
        beat_times = np.cumsum(final_rr)
        beat_times = np.insert(beat_times, 0, 0.0) # 起始點
        
        # 為了產生連續的頻率曲線，我們做簡單的階梯函數或插值
        # 頻率 = 1 / RR
        hr_values = 1.0 / np.array(final_rr)
        
        # 建立時間軸
        t_total = np.linspace(0, duration, int(duration * self.fs))
        
        # 使用插值法將「離散的心率」擴展到「連續的時間軸」
        # 'previous' 模式代表在下一次心跳來之前，頻率維持不變 (Step Function)
        # 這能確保波形的寬度在該次心跳中保持固定
        f_interp = interp1d(beat_times[:-1], hr_values, kind='previous', fill_value="extrapolate")
        hr_freq_array = f_interp(t_total)
        
        # 5. 積分得到相位
        theta = 2 * np.pi * np.cumsum(hr_freq_array) / self.fs
        
        return theta, hr_freq_array, t_total

    def get_random_bcg_params(self, hr_freq): 
        """ 自動生成一組隨機變異的 McSharry 參數。 策略：以一組「標準參數」為基底，進行隨機縮放與微擾。 """ 

        std_a = np.array([0.3, -0.7, 1.2, -0.9, 0.5]) 
        std_b = np.array([0.3, 0.24, 0.4, 0.3, 0.4]) 
        std_t = np.array([-1.4, -0.6, 0.0, 0.6, 1.8]) 

        width_scale = np.sqrt(1/hr_freq)
        if hr_freq.ndim > 0:
            width_scale = width_scale[np.newaxis, :] 
            new_b = std_b[:, np.newaxis] * width_scale 
            new_t = std_t[:, np.newaxis] * width_scale 
        else:
            new_b = std_b * width_scale
            new_t = std_t * width_scale
        # new_b = std_b * width_scale 
        # new_t = std_t * width_scale 
        amp_noise = np.random.uniform(0.8, 1.2, size=5) 
        new_a = std_a #* amp_noise 
        pos_jitter = np.random.uniform(-0.01, 0.01, size=5) 
        pos_jitter = np.random.uniform(-0.02, 0.02, size=(5, hr_freq.shape[0]))
        new_t = new_t + pos_jitter
        new_t[2] = 0.0 # Lock J wave
        
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
        # 產生相位與總心率 (混合了 RSA + Intrinsic HRV)
        # theta, hr_mean, t = self.get_total_phase(duration, fs=100, mean_bpm=60, rsa_signal=resp_signal, hrv_scale=0.8)
        theta, hr_mean, t = self.generate_arrhythmia_profile(duration, mean_bpm=60, anomaly_prob=0.15)
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