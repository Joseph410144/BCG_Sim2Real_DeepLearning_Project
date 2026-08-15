import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks, resample
from Algorithm.Data_pre_processing import zscore_normalize, regular, Smoothmethod
from Algorithm.Filters import BandPassFilter


def compute_bcg_heart_psd(signal, heartrate, bcg_sf):
    """ calculate heart frequency """
    heart_frequency = heartrate/60
    """ fourier transform """
    Nfft = len(signal)
    bcg_fft = fft(signal)
    Frequency_index = fftfreq(Nfft, 1/bcg_sf)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    bcg_fft = np.abs(bcg_fft[mask])
    low_frequency_0p7 = Frequency_index <= 0.7
    bcg_fft[low_frequency_0p7] = 0
    """ get 0-3Hz spectrum """
    low_frequency = Frequency_index <= 3
    bcg_fft = bcg_fft[low_frequency][1:]
    Frequency_index = Frequency_index[low_frequency][1:]

    psd_sum_heart_frequency = 0
    index = 1
    # print(f'heart freq: {heart_frequency}')
    # while index*heart_frequency+0.25 < Frequency_index[-1]:
    for index in range(1, 3):
        iter_frequency = index*heart_frequency
        if iter_frequency+0.2 > Frequency_index[-1]:
            break
        indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(iter_frequency-0.2))))[0]
        indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(iter_frequency+0.2))))[0]
        # print(f'sum: {Frequency_index[indexF1]}, {Frequency_index[indexF2]}')
        if indexF1 and indexF2:
            psd_sum_heart_frequency = psd_sum_heart_frequency + sum(bcg_fft[indexF1[0]:indexF2[0]])
        index += 1

    return psd_sum_heart_frequency/sum(bcg_fft)

def MCUV4(BCGSignal, SavePathMin):
    BCGSignal = np.array(BCGSignal)
    BCGsampleFrequency = 100

    """ Frequency Convolution vs Time dot  """
    """ Band pass filter """
    BcgData = np.array(BCGSignal)
    BcgData = BandPassFilter(BcgData, 0.7, 3, 4, BCGsampleFrequency)
    """ fourier transform """
    BCGfs = 100
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 10
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        # index = 1
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            # while index*closetF<5:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.02))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.02))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])
                # index += 1

            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.02))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.02))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
    else:
        MajorFreqA = 2*maxF

    SaveAllTxt = os.path.join(SavePathMin, "Frequency compare.txt")

    with open(SaveAllTxt, 'w') as f:
        f.write(f'fundamental frequency: {maxF}, amplitude: {AmF1}\n2 times frequency: {2*maxF}, amplitude: {AmF2}\n')

    plt.plot(ansF, ansA)
    plt.title("BCG FFT PSD")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("amplitude(accumulation)")
    plt.savefig(os.path.join(SavePathMin, f"BCG_FFT_PSD_V4.png"))
    plt.close()

    plt.plot(Frequency_index[Frequency_index<=3], regular(BCG_fft_BcgData[Frequency_index<=3]))
    plt.text(1.8, 0.9, f'Main frequency: {round(MajorFreqA, 2)}')
    plt.title("BCG FFT threshold")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("amplitude(normalize)")
    plt.savefig(os.path.join(SavePathMin, f"BCG_FFTthreshold_Ani_V4.png"))
    plt.close()

    FilteredBCGSignal = BandPassFilter(BCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGsampleFrequency)
    FilteredBCGSignal = BandPassFilter(FilteredBCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGsampleFrequency)
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))

    return JJi, FilteredBCGSignal, Peaks, BCG_fft_BcgData[Frequency_index<=3]

def MCUV5(BCGSignal, SavePathMin):
    BCGSignal = np.array(BCGSignal)
    BCGsampleFrequency = 100

    """ Band pass filter """
    BcgData = np.array(BCGSignal)
    BCG__ab = BcgData.copy()
    BcgData = BandPassFilter(BcgData, 2, 5, 4, BCGsampleFrequency)
    """ signal square """
    BcgData = np.square(BcgData)
    """ fourier transform """
    BCGfs = 100
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 3
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.02))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.02))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])
            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.02))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.02))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
    else:
        MajorFreqA = 2*maxF

    plt.plot(ansF, ansA)
    plt.title("BCG FFT PSD")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("amplitude(accumulation)")
    plt.savefig(os.path.join(SavePathMin, f"BCG_FFT_PSD_V5.png"))
    plt.close()

    plt.plot(Frequency_index[Frequency_index<=3], regular(BCG_fft_BcgData[Frequency_index<=3]))
    plt.text(1.8, 0.9, f'Main frequency: {round(MajorFreqA, 2)}')
    plt.title("BCG FFT threshold")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("amplitude(normalize)")
    plt.savefig(os.path.join(SavePathMin, f"BCG_FFTthreshold_Ani_V5.png"))
    plt.close()

    FilteredBCGSignal = BandPassFilter(BCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGsampleFrequency)
    FilteredBCGSignal = BandPassFilter(FilteredBCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGsampleFrequency)
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))

    return JJi, FilteredBCGSignal, Peaks, BCG_fft_BcgData[Frequency_index<=3]

def MCUV4_TenSecond(BCGSignal):
    BcgData = np.array(BCGSignal)
    BCGsampleFrequency = 100
    """ BP 0.5-25 """
    # paddingLength = 50
    # xx = np.concatenate((np.array(BCGSignal)[:paddingLength][::-1], BCGSignal, np.array(BCGSignal)[-paddingLength:][::-1]))
    # BcgData = BandPassFilter(xx, 0.5, 25, 4, 100)[paddingLength:-paddingLength]

    """ Frequency Convolution vs Time dot  """
    """ Band pass filter """
    paddingLength = 50
    xx = np.concatenate((np.array(BcgData)[:paddingLength][::-1], BcgData, np.array(BcgData)[-paddingLength:][::-1]))
    BcgData = BandPassFilter(xx, 0.7, 3, 4, BCGsampleFrequency)[paddingLength:-paddingLength]
    """ fourier transform """
    BCGfs = 100
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 10
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.1))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.1))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.1))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.1))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.1))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.1))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
    else:
        MajorFreqA = 2*maxF

    xx = np.concatenate((np.array(BCGSignal)[:paddingLength][::-1], BCGSignal, np.array(BCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGsampleFrequency)[paddingLength:-paddingLength]
    xx = np.concatenate((np.array(FilteredBCGSignal)[:paddingLength][::-1], FilteredBCGSignal, np.array(FilteredBCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGsampleFrequency)[paddingLength:-paddingLength]
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))

    return round((60000/np.mean(np.array(JJi)))/1000, 2)

def MCUV5_TenSecond(BCGSignal):
    BCGSignal = np.array(BCGSignal)
    BCGfs = 100
    """ Frequency Convolution vs Time dot  """
    """ Band pass filter """
    paddingLength = 50
    BCG__ab = BCGSignal.copy()
    BcgData = BandPassFilter(BCGSignal, 2, 5, 4, BCGfs)
    """ signal square """
    # BcgData = np.diff(BcgData)
    BcgData = np.abs(BcgData)
    """ fourier transform """
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 3
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.1))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.1))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.2))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.2))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.2))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.2))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
        ansAM = AmF1
    else:
        MajorFreqA = 2*maxF
        ansAM = AmF2

    xx = np.concatenate((np.array(BCGSignal)[:paddingLength][::-1], BCGSignal, np.array(BCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGfs)[paddingLength:-paddingLength]
    xx = np.concatenate((np.array(FilteredBCGSignal)[:paddingLength][::-1], FilteredBCGSignal, np.array(FilteredBCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGfs)[paddingLength:-paddingLength]
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGfs))

    return round((60000/np.mean(np.array(JJi)))/1000, 2)

def MCUV4_R(BCGSignal):
    BCGSignal = np.array(BCGSignal)
    BCGsampleFrequency = 100

    """ Frequency Convolution vs Time dot  """
    """ Band pass filter """
    BcgData = np.array(BCGSignal)
    BcgData = BandPassFilter(BcgData, 0.7, 3, 4, BCGsampleFrequency)
    """ fourier transform """
    BCGfs = 100
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 10
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.02))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.02))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])
            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.02))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.02))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
    else:
        MajorFreqA = 2*maxF

    FilteredBCGSignal = BandPassFilter(BCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGsampleFrequency)
    FilteredBCGSignal = BandPassFilter(FilteredBCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGsampleFrequency)
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))

    return JJi

def MCUV5_R(BCGSignal):
    BCGSignal = np.array(BCGSignal)
    BCGsampleFrequency = 100

    """ Band pass filter """
    BcgData = np.array(BCGSignal)
    BCG__ab = BcgData.copy()
    BcgData = BandPassFilter(BcgData, 2, 5, 4, BCGsampleFrequency)
    """ signal square """
    BcgData = np.diff(BcgData)
    BcgData = np.abs(BcgData)
    """ fourier transform """
    BCGfs = 100
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 3
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.02))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.02))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])
            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.02))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.02))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.02))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
    else:
        MajorFreqA = 2*maxF

    FilteredBCGSignal = BandPassFilter(BCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGsampleFrequency)
    FilteredBCGSignal = BandPassFilter(FilteredBCGSignal, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGsampleFrequency)
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))

    return JJi

def MCUV5_TenSecond(BCGSignal):
    BCGSignal = np.array(BCGSignal)
    BCGfs = 100
    """ Frequency Convolution vs Time dot  """
    """ Band pass filter """
    paddingLength = 50
    BCG__ab = BCGSignal.copy()
    BcgData = BandPassFilter(BCGSignal, 2, 5, 4, BCGfs)
    """ signal square """
    # BcgData = np.diff(BcgData)
    BcgData = np.abs(BcgData)
    """ fourier transform """
    Nfft = len(BcgData)
    BCG_fft_BcgData = fft(BcgData)

    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = np.abs(BCG_fft_BcgData[mask])

    low_frequency_10 = Frequency_index <= 3
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]

    ansF = []
    ansA = []
    for i in range(1, len(BCG_fft_BcgData)):
        ans = 0
        closetF = Frequency_index[i]
        ansF.append(closetF)
        if closetF > 0.8:
            for index in range(1, 3):
                midClosetf = index*closetF
                indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf-0.1))))[0]
                indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(midClosetf+0.1))))[0]
                if indexF1 and indexF2:
                    ans = ans + sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

            if ans:
                ansA.append(ans)
            else:
                ansA.append(0)
        else:
            ansA.append(0)

    aa = np.where(ansA == max(ansA))[0]
    maxF = ansF[aa[0]]
    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF-0.2))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(maxF+0.2))))[0]
    AmF1 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    indexF1 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF-0.2))))[0]
    indexF2 = np.where(Frequency_index == min(Frequency_index, key=lambda x: abs(x-(2*maxF+0.2))))[0]
    AmF2 = sum(BCG_fft_BcgData[indexF1[0]:indexF2[0]])

    if 2.5*AmF1 >= AmF2:
        MajorFreqA = maxF
        ansAM = AmF1
    else:
        MajorFreqA = 2*maxF
        ansAM = AmF2

    xx = np.concatenate((np.array(BCGSignal)[:paddingLength][::-1], BCGSignal, np.array(BCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 1, BCGfs)[paddingLength:-paddingLength]
    xx = np.concatenate((np.array(FilteredBCGSignal)[:paddingLength][::-1], FilteredBCGSignal, np.array(FilteredBCGSignal)[-paddingLength:][::-1]))
    FilteredBCGSignal = BandPassFilter(xx, MajorFreqA-0.2, MajorFreqA+0.2, 2, BCGfs)[paddingLength:-paddingLength]
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGfs))

    return round((60000/np.mean(np.array(JJi)))/1000, 2)

def HeartV6_TenSecond(BCGSignal):
    BCGSignal = np.array(BCGSignal)
    BCGsampleFrequency = 100
    """ Band pass filter """
    RawData = np.array(BCGSignal)
    BCG__ab = RawData.copy()
    BCG_fft_BcgData = np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 1, 5, 4, 100)))) \
                    + np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 2, 6, 4, 100)))) \
                    + np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 3, 7, 4, 100)))) \
                    + np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 4, 8, 4, 100)))) \
                    + np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 5, 9, 4, 100)))) \
                    + np.abs(np.fft.fft(np.abs(BandPassFilter(RawData, 6, 10, 4, 100))))

    """ fourier transform """
    BCGfs = 100
    Nfft = len(RawData)
    # BCG_fft_BcgData = fft(BcgData)
    Frequency_index = fftfreq(Nfft, 1/BCGfs)
    mask = Frequency_index >= 0
    Frequency_index = Frequency_index[mask]
    BCG_fft_BcgData = BCG_fft_BcgData[mask]

    low_frequency_10 = Frequency_index <= 3
    BCG_fft_BcgData = BCG_fft_BcgData[low_frequency_10][1:]
    Frequency_index = Frequency_index[low_frequency_10][1:]
    MajorFreqA = PSD_ratio_heart_freq_alg_V2_10sec(Frequency_index, BCG_fft_BcgData)

    FilteredBCGSignal = BandPassFilter(BCGSignal, MajorFreqA-0.15, MajorFreqA+0.15, 1, BCGsampleFrequency)
    FilteredBCGSignal = BandPassFilter(FilteredBCGSignal, MajorFreqA-0.15, MajorFreqA+0.15, 2, BCGsampleFrequency)
    Peaks, _ = find_peaks(FilteredBCGSignal, height=0)
    JJi = []
    for i in range(1, len(Peaks)):
        JJi.append(((Peaks[i]-Peaks[i-1])/BCGsampleFrequency))
    bcg_jji = np.array(JJi)
    bcg_heartrate_alg = round(60/np.mean(np.array(bcg_jji)))

    return bcg_heartrate_alg

def PSD_ratio_heart_freq_alg_V2_10sec(Frequency_index, BCG_fft_BcgData):
    # consider heart frequency and modulation frequency
    heart_frequency = 0
    heart_frequency_energy = 0
    heart_frequency_lists = []
    heart_frequency_energy_lists = []
    for i in range(1, len(BCG_fft_BcgData)):
        frequency = Frequency_index[i]
        frequency_energy = 0
        index_multi = 1
        if frequency > 0.7:
            while index_multi*frequency<3:
                target_freq = frequency*index_multi
                for j in range(len(Frequency_index)):
                    if Frequency_index[j]>=target_freq-0.15 and Frequency_index[j]<=target_freq+0.15:
                        frequency_energy += BCG_fft_BcgData[j]
                index_multi += 1
        # normalize frequency energy
        frequency_energy = frequency_energy/index_multi
        # print(f'iter frequency: {frequency}, amp: {frequency_energy}, heart energy: {heart_frequency_energy}')
        heart_frequency_lists.append(frequency)
        heart_frequency_energy_lists.append(frequency_energy)
        if frequency_energy > heart_frequency_energy:
            heart_frequency = frequency
            heart_frequency_energy = frequency_energy
            # print(heart_frequency, heart_frequency_energy)


    # plt.plot(heart_frequency_lists, heart_frequency_energy_lists)
    # plt.title("BCG FFT PSD")
    # plt.xlabel("Frequency(Hz)")
    # plt.ylabel("amplitude(accumulation)")
    # plt.show()
    # return heart_frequency
    index_multi = 1
    basic_frequency = heart_frequency
    max_frequency = 0
    max_frequency_energy = 0
    while index_multi*basic_frequency<3:
        frequency_iter = index_multi*basic_frequency
        frequency_iter_energy = 0
        for j in range(len(Frequency_index)):
            if Frequency_index[j]>=frequency_iter-0.15 and Frequency_index[j]<=frequency_iter+0.15:
                frequency_iter_energy += BCG_fft_BcgData[j]
        if index_multi == 1:
            frequency_iter_energy = frequency_iter_energy * 2.5
        if frequency_iter_energy > max_frequency_energy:
            max_frequency_energy = frequency_iter_energy
            max_frequency = frequency_iter

        index_multi += 1

    return max_frequency