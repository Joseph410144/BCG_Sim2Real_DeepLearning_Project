import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from Model import LSTM
from Algorithm.Data_pre_processing import zscore_normalize, BandPassFilter
from Algorithm.BCG_heartrate_alg import MCUV5_TenSecond
from Algorithm.ECG_heartrate_alg import DetectionECGPeaks_TenSond


def fft_loss(pred_bcg, target_bcg, target_ecg, fs=100, band=10, w_time=0.2, w_fft=1, eps=1e-7):
    # pred,target: [B,C,T]
    mse_time = torch.mean((pred_bcg - target_ecg)**2)
    original_fft = torch.abs(torch.fft.rfft(torch.abs(target_bcg), dim=-1))  # [B,C,F]
    pred_fft = torch.abs(torch.fft.rfft(pred_bcg, dim=-1))  # [B,C,F]
    target_fft = torch.abs(torch.fft.rfft(target_ecg, dim=-1))

    freqs = torch.fft.rfftfreq(pred_bcg.shape[-1], d=1.0/fs).to(pred_bcg.device)
    band_mask = freqs <= band
    band_mask = band_mask
    
    original_fft = original_fft[:, :, band_mask]
    pred_fft = pred_fft[:, :, band_mask]
    target_fft = target_fft[:, :, band_mask]
    freqs = freqs[band_mask]

    pred_fft = (pred_fft - pred_fft.mean(dim=-1, keepdim=True)) / pred_fft.std(dim=-1, keepdim=True)
    target_fft = (target_fft - target_fft.mean(dim=-1, keepdim=True)) / target_fft.std(dim=-1, keepdim=True)
    
    
    mse_fft = torch.mean((pred_fft - target_fft)**2)
    # print('time, frequency loss', mse_time.item(), mse_fft.item())

    return freqs.squeeze().cpu().detach().numpy()[1:], zscore_normalize(original_fft[0][0].squeeze().cpu().detach().numpy()[1:]), zscore_normalize(pred_fft[0][0].squeeze().cpu().detach().numpy()[1:]), zscore_normalize(target_fft[0][0].squeeze().cpu().detach().numpy()[1:])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    data_length = 1000
    """ testing data path """
    problem_data_path = r'/Users/joseph/Documents/Program/Dataset/BCG/A_ballistocardiogram_dataset_with_reference_ECG_signals_for_bedside_heart_rhythm assessment/PreProcessed_dataset/subject02'
    
    """ model """
    pre_model = LSTM.LSTM_BCGFilter_Pre(seq_len=data_length, input_size=1, hidden_size=128, output_size=1,
            dropout=0.2, num_layers=6, bidirectional=True, repeat_times = 1)
    pre_model.to(DEVICE)
    pre_model.load_state_dict(torch.load(r'/Users/joseph/Documents/Program/BCG_DeepLearning/weight/BCG_HeartFilter/260109/best_Test_model.pth', map_location=DEVICE))
    pre_model.eval()

    # pre_model = LSTM.LSTM_BCGFilter_Pre_Confidence(seq_len=data_length, input_size=1, hidden_size=128, output_size=1,
    #         dropout=0.2, num_layers=6, bidirectional=True, repeat_times = 1)
    # pre_model.to(DEVICE)
    # pre_model.load_state_dict(torch.load(r'D:\Program\innolux_BCG_deep_learning\weight\BCG_HeartFilter\251231\best_Test_model.pth', map_location=DEVICE))
    # pre_model.eval()

    problems = 0
    for file in os.listdir(problem_data_path):
        # bcg_heart_signal_1, ecg_signal_1, _ = np.load(os.path.join(problem_data_path, file))
        bcg_heart_signal_1, ecg_signal_1 = np.load(os.path.join(problem_data_path, file))
        bcg_heart_signal_1 = BandPassFilter(signal_=bcg_heart_signal_1, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500)
        ecg_signal_1 = BandPassFilter(signal_=ecg_signal_1, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500)
        bcg_heart_signal_1 = zscore_normalize(bcg_heart_signal_1)[:data_length]
        # ecg_signal_1 = BandPassFilter(signal_=ecg_signal_1, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500)
        ecg_signal_1 = zscore_normalize(ecg_signal_1)[:data_length]
        BCG_tensor = torch.tensor(bcg_heart_signal_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1000]
        ECG_tensor = torch.tensor(ecg_signal_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1000]
        predicted = pre_model(BCG_tensor.to(DEVICE))  # output shape: [1, 1, 1000]

        freqs, original_bcg_spectrum, pecg_spectrum, ecg_spectrum = fft_loss(predicted, BCG_tensor.to(DEVICE), ECG_tensor.to(DEVICE))
        
        # plt.figure()
        # plt.subplot(311)
        # plt.plot(freqs, original_bcg_spectrum, label='original BCG spectrum')
        # plt.legend(loc='upper left')
        # plt.subplot(312)
        # plt.plot(freqs, pecg_spectrum, label='Model De-noising BCG spectrum')
        # plt.legend(loc='upper left')
        # plt.subplot(313)
        # plt.plot(freqs, ecg_spectrum, label='original ECG spectrum')
        # plt.xlabel('Frequency')
        # plt.legend(loc='upper left')

        # fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 2)
        # ax_1.plot(bcg_heart_signal_1)
        # ax_1.set_title('Original BCG time domain')
        # ax_2.plot(predicted.squeeze().cpu().detach().numpy())
        # # ax_2.set_title(f'pECG time domain, confidence socre: {round(float(probability.squeeze().cpu().detach().numpy()), 2)}')
        # ax_2.set_title(f'Model De-noising BCG time domain')
        # ax_3.plot(ecg_signal_1)
        # ax_3.set_title('ECG time domain')
        # ax_3.set_xlabel('time(points)')
        # plt.show()

        # 設定畫布大小 (寬一點，讓左右並排不擠)
        fig, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)
        
        # --- Row 1: Original Input ---
        # 左邊: Time
        axes[0, 0].plot(bcg_heart_signal_1, color='#1f77b4', linewidth=1.2) # 藍色
        axes[0, 0].set_title('(A) Raw Real-World BCG (Time)', loc='left', fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        
        # 右邊: Freq
        axes[0, 1].plot(freqs, original_bcg_spectrum, color='#1f77b4', linewidth=1.2)
        axes[0, 1].set_title('(B) Raw BCG Spectrum (Freq)', loc='left', fontweight='bold')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)

        # --- Row 2: Model Output ---
        # 左邊: Time
        axes[1, 0].plot(predicted.squeeze().cpu().detach().numpy(), color='#d62728', linewidth=1.5) # 紅色
        axes[1, 0].set_title('(C) Model De-noised BCG (Time)', loc='left', fontweight='bold')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        
        # 右邊: Freq
        axes[1, 1].plot(freqs, pecg_spectrum, color='#d62728', linewidth=1.5)
        axes[1, 1].set_title('(D) De-noised BCG Spectrum (Freq)', loc='left', fontweight='bold')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)

        # --- Row 3: Ground Truth ---
        # 左邊: Time
        axes[2, 0].plot(ecg_signal_1, color='black', linewidth=1.5) # 黑色
        axes[2, 0].set_title('(E) Reference ECG (Time)', loc='left', fontweight='bold')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].set_xlabel('Time (points)')
        axes[2, 0].grid(True, linestyle=':', alpha=0.6)
        
        # 右邊: Freq
        axes[2, 1].plot(freqs, ecg_spectrum, color='black', linewidth=1.5)
        axes[2, 1].set_title('(F) Original ECG Spectrum (Freq)', loc='left', fontweight='bold')
        axes[2, 1].set_ylabel('Magnitude')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].grid(True, linestyle=':', alpha=0.6)
        
        # 統一 X 軸範圍 (美觀)
        # Time Domain X軸對齊
        axes[0, 0].set_xlim(0, len(bcg_heart_signal_1))
        axes[1, 0].set_xlim(0, len(bcg_heart_signal_1))
        axes[2, 0].set_xlim(0, len(bcg_heart_signal_1))
        
        # Freq Domain X軸對齊 (例如只看 0-10Hz)
        axes[0, 1].set_xlim(0, 10)
        axes[1, 1].set_xlim(0, 10)
        axes[2, 1].set_xlim(0, 10)

        # 存檔
        # plt.savefig('fig2.jpg', dpi=300)
        plt.show()
    # print(problems)
            