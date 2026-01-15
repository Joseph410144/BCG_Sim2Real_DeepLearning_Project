import torch
import numpy as np

from scipy import signal

def zscore_normalize(x):
    """
    用來做標準化的函式
    Input: 一段訊號
    Output: 被標準化後的訊號(資料總長度不變)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()  # 確保轉成 numpy
    mu = np.mean(x)
    sigma = np.std(x)
    x_norm = (x-mu)/sigma
    return x_norm

def regular(x):
    """
    用來做正規化的函式
    Input: 一段訊號
    Output: 被正規化後的訊號(資料總長度不變)
    """
    min_x = np.min(x)
    max_x = np.max(x)
    x_re = (x-min_x)/(max_x-min_x)
    return x_re

def Smoothmethod(data, method, point):
  """ 
  用來做平滑的函式
  Input: 一段訊號和平滑方法和一次要用多少個點做平滑
  Output: 平滑完的訊號
  """
  SmoothData = []
  if method=="EMA":
    dis = point//2
    for i in range(len(data)):
      if i-dis<0:
        left = 0
      else:
        left = i-dis
      if i+dis>len(data)-1:
        right = len(data)-1
      else:
        right = i+dis

      SmoothData.append(np.mean(data[left:right]))

  return SmoothData