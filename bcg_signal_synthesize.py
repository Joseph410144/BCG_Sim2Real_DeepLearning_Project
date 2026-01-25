import os, random
import numpy as np
import matplotlib.pyplot as plt

from Algorithm.Filters import BandPassFilter
from Algorithm.Bcg_signal_synthesis_function import BedBCGGenerator, BedBCGGenerator_V3

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
        amp_n = (heart_amp_base/n) * random.uniform(0.8, 1.2)
        # 隨機相位 (關鍵！讓波形形狀每次都不同)
        phi_n = random.uniform(0, 2*np.pi)
        
        # 使用 theta_h (含 RSA)，這樣所有諧波都會一起忽快忽慢
        heart_signal += amp_n * np.cos(n * theta_h + phi_n)
        
    # 正規化心跳振幅 (讓 Deep Learning 比較好學)
    heart_signal = heart_signal / np.std(heart_signal)
        
    return heart_signal

if __name__ == '__main__':
    # save path
    data_number = 100000
    training_path = '/Users/joseph/Documents/Program/Dataset/BCG/Synthesis/training'
    val_path = '/Users/joseph/Documents/Program/Dataset/BCG/Synthesis/validation'
    test_path = '/Users/joseph/Documents/Program/Dataset/BCG/Synthesis/test'
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    generate_phase = {'training':(training_path, (data_number*8)//10), 'validation':(val_path, (data_number*2)//10), 'testing':(test_path, (data_number*1)//10)}

    # signal parameter
    signal_time = 10
    sampling_rate = 100
    t = np.linspace(0, signal_time, signal_time * sampling_rate)

    # generator
    bcg_generator = BedBCGGenerator(fs=sampling_rate)
    
    # 雜訊參數
    mean = 0
    std_dev = 0.2
    size = len(t)
    
    for phase in generate_phase.keys():
        print(f'start generating {phase} data')
        save_path = generate_phase[phase][0]
        for i in range(generate_phase[phase][1]):
            # BCG generating V2
            noisy_bcg, clean_bcg = bcg_generator.generate(duration=10)
            synthesis_model_data = np.array([noisy_bcg, clean_bcg])
            np.save(os.path.join(save_path, f'{phase}_data_{i}.npy'), synthesis_model_data)

