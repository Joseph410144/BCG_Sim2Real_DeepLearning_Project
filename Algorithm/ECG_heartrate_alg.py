import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Algorithm.Data_pre_processing import Smoothmethod
from scipy.stats import norm

def DetectionECGPeaks(XEcg, k, p, sfreq):
    Xdet = Smoothmethod(XEcg, "EMA", 10)#np.convolve(XEcg, np.ones((40,))/40, mode="same")

    """ Get Xsqr """
    Xdet_mu = np.mean(Xdet[:])
    Xdet_std = np.std(Xdet[:])
    Xsqr = []
    for i in XEcg:
        if i >= (Xdet_mu+k*Xdet_std):
            Xsqr.append(1)
        else:
            Xsqr.append(-1)
    Xsqr = np.array(Xsqr)

    """ Get Positive Intervals """
    pos_int = False
    posStart = []
    posEnd = []
    for i in range(len(Xsqr)):
        if Xsqr[i]>0 and (not pos_int):
            posStart.append(i)
            pos_int = True
        elif Xsqr[i]<0 and pos_int:
            posEnd.append(i)
            pos_int = False

        if i==len(Xsqr)-1 and pos_int:
            posEnd.append(i)
            pos_int = False
    for update in range(50):
        posStartAd = []
        posEndAd = []
        AnomalyNum = 0
        previous = False
        for i in range(len(posStart)):
            if previous:
                previous = False
                continue
            if i != len(posStart)-1:
                if ((posStart[i+1]+posEnd[i+1])/2)/sfreq-((posEnd[i]+posStart[i])/2)/sfreq <= 0.5:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i+1])
                    AnomalyNum+=1
                    previous = True
                else:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])
            if i == len(posStart)-1:
                if not previous:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])

        if AnomalyNum > 240:
            break
        else:
            posStart = posStartAd
            posEnd = posEndAd

    """ Find J peaks """
    Peaks = []

    for i in range(0, len(posStart)):
        if posStart[i]==posEnd[i]:
            continue
        p = np.argmax(XEcg[posStart[i]:posEnd[i]])
        Peaks.append(p+posStart[i])

    RRinterval = 0
    RRi = []
    num = 0
    for i in range(0, len(Peaks)):
        if i==0:
            continue
        else:
            num+=1
            RRinterval += ((Peaks[i]-Peaks[i-1])/sfreq)
            RRi.append(((Peaks[i]-Peaks[i-1])/sfreq))
    
    # 回傳JJ interval的矩陣，找到的心跳峰值，還有尋找j peak的區間
    return RRi, Peaks, Xsqr

def DetectionECGPeaks_TenSond(XEcg, k, p, sfreq):

    Xdet = Smoothmethod(XEcg, "EMA", 10)#np.convolve(XEcg, np.ones((40,))/40, mode="same")

    """ Get Xsqr """
    # 找出可出現心跳peak的區間和計算閥值
    Xdet_mu = np.mean(Xdet[p*-1:])
    Xdet_std = np.std(Xdet[p*-1:])
    Xsqr = []
    for i in XEcg:
        if i >= (Xdet_mu+k*Xdet_std):
            Xsqr.append(1)
        else:
            Xsqr.append(-1)
    Xsqr = np.array(Xsqr)

    """ Get Positive Intervals """
    pos_int = False
    posStart = []
    posEnd = []
    for i in range(len(Xsqr)):
        if Xsqr[i]>0 and (not pos_int):
            posStart.append(i)
            pos_int = True
        elif Xsqr[i]<0 and pos_int:
            posEnd.append(i)
            pos_int = False

        if i==len(Xsqr)-1 and pos_int:
            posEnd.append(i)
            pos_int = False
            
    for update in range(50):
        posStartAd = []
        posEndAd = []
        AnomalyNum = 0
        previous = False
        for i in range(len(posStart)):
            if previous:
                previous = False
                continue
            if i != len(posStart)-1:
                if ((posStart[i+1]+posEnd[i+1])/2)/sfreq-((posEnd[i]+posStart[i])/2)/sfreq <= 0.5:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i+1])
                    AnomalyNum+=1
                    previous = True
                else:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])
            if i == len(posStart)-1:
                if not previous:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])

        if AnomalyNum > 240:
            break
        else:
            posStart = posStartAd
            posEnd = posEndAd

    """ Find J peaks """
    Peaks = []

    for i in range(0, len(posStart)):
        if posStart[i]==posEnd[i]:
            continue
        p = np.argmax(XEcg[posStart[i]:posEnd[i]])
        Peaks.append(p+posStart[i])
    # plt.plot(XEcg)
    # plt.plot(Peaks, XEcg[Peaks], 'o')
    # plt.show()

    RRinterval = 0
    RRi = []
    num = 0
    for i in range(0, len(Peaks)):
        if i==0:
            continue
        else:
            num+=1
            RRinterval += ((Peaks[i]-Peaks[i-1])/sfreq)
            RRi.append(((Peaks[i]-Peaks[i-1])/sfreq))

    # 回傳JJ interval的矩陣，找到的心跳峰值，還有尋找j peak的區間
    return round((60000/np.mean(np.array(RRi)))/1000, 2)

def Detection_ECG_resp_peaks(ECG_resp_signal, SavePath):
    """ do not need another low pass filter """
    RespData = ECG_resp_signal
    # parameter k to decide threshold line
    k = 0.5
    Resp_mu = np.mean(RespData)
    Resp_std = np.std(RespData)
    threshold = Resp_mu+k*Resp_std

    """ get peaks intervals """
    Xsqr = []
    for i in RespData:
        if i >= (threshold):
            Xsqr.append(1)
        else:
            Xsqr.append(-1)
    pos_int = False
    posStart = []
    posEnd = []
    for i in range(len(Xsqr)):
        if Xsqr[i]>0 and (not pos_int):
            posStart.append(i)
            pos_int = True
        elif Xsqr[i]<0 and pos_int:
            posEnd.append(i)
            pos_int = False

        if i==len(Xsqr)-1 and pos_int:
            posEnd.append(i)
            pos_int = False
    
    Peaks = []

    for i in range(0, len(posStart)):
        if posStart[i]==posEnd[i]:
            continue
        p = np.argmax(RespData[posStart[i]:posEnd[i]])
        Peaks.append(p+posStart[i])
    
    if len(Peaks) <= 1:
        return -1, -1

    # peaks_new = []
    # peaks_mean, peaks_std = np.mean(RespData[Peaks]), np.std(RespData[Peaks])
    # for p in Peaks:
    #     if RespData[p] > peaks_mean - peaks_std:
    #         peaks_new.append(p)
    
    # Peaks = peaks_new

    plt.figure(figsize=(20, 3))
    plt.plot(RespData)
    plt.plot(Peaks, RespData[Peaks], 'o')
    plt.axhline(y=threshold, color='r', linestyle="--")
    plt.grid()
    plt.yticks(fontsize=10)
    plt.title("Resp Peaks")
    plt.savefig(os.path.join(SavePath, f"ECG_resp_waveform_V3_Peaks.png"))
    plt.close()

    RespInterval = []
    for i in range(1, len(Peaks)):
        RespInterval.append(((Peaks[i]-Peaks[i-1])/50))
    
    AverageAmplitude = np.mean(RespData[Peaks])
    
    return round((60/np.mean(np.array(RespInterval))), 0)

def ECG_R_peak_weight(XEcg, k, p, sfreq):
    """ 
    用偵測訊號裡面peak的函式
    Input: 心跳訊號、參數k、參數p、positive interval的更新次數、是偵測呼吸還是心跳訊號、採樣頻率
    Output: JJI, Peaks的位置, Xsqr
    """

    """ if want to detect Breathe Peaks >> k=1 ; heart peaks >> k=0.35"""

    """ Parameters """
    Xdet = Smoothmethod(XEcg, "EMA", 10)#np.convolve(XEcg, np.ones((40,))/40, mode="same")

    """ Get Xsqr """
    # 找出可出現心跳peak的區間和計算閥值
    Xdet_mu = np.mean(Xdet[:])
    Xdet_std = np.std(Xdet[:])
    Xsqr = []
    for i in XEcg:
        if i >= (Xdet_mu+k*Xdet_std):
            Xsqr.append(1)
        else:
            Xsqr.append(-1)
    Xsqr = np.array(Xsqr)


    """ Get Positive Intervals """
    pos_int = False
    posStart = []
    posEnd = []
    for i in range(len(Xsqr)):
        if Xsqr[i]>0 and (not pos_int):
            posStart.append(i)
            pos_int = True
        elif Xsqr[i]<0 and pos_int:
            posEnd.append(i)
            pos_int = False

        if i==len(Xsqr)-1 and pos_int:
            posEnd.append(i)
            pos_int = False
    for update in range(50):
        posStartAd = []
        posEndAd = []
        AnomalyNum = 0
        previous = False
        for i in range(len(posStart)):
            if previous:
                previous = False
                continue
            if i != len(posStart)-1:
                if ((posStart[i+1]+posEnd[i+1])/2)/sfreq-((posEnd[i]+posStart[i])/2)/sfreq <= 0.5:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i+1])
                    AnomalyNum+=1
                    previous = True
                else:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])
            if i == len(posStart)-1:
                if not previous:
                    posStartAd.append(posStart[i])
                    posEndAd.append(posEnd[i])

        if AnomalyNum > 240:
            break
        else:
            posStart = posStartAd
            posEnd = posEndAd

    # 建立 1000 點的全 0 陣列
    x = np.zeros(1000)

    # 找出每段區間的最大值索引（J peak）
    peaks = [np.argmax(XEcg[start:end]) + start for start, end in zip(posStart, posEnd) if start != end]
    radius = 30
    length = 1000

    # 權重從中間 1 漸進式下降到 0.2，可調整
    weights = np.linspace(0.2, 1.0, radius + 1)
    weights = np.concatenate([weights[:-1], weights[::-1]])  # 對稱排列，中心為 1

    mask = np.zeros(length)

    for p in peaks:
        for i, w in enumerate(weights):
            idx = p + i - radius
            if 0 <= idx < length:
                mask[idx] = max(mask[idx], w)  # 保留最大權重，避免重疊變小

    return mask