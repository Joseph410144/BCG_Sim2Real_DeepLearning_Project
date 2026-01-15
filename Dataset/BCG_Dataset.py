import os
import torch
import numpy as np

from scipy import signal
from torch.utils.data import Dataset
from scipy.fft import fft, ifft, fftfreq
from Algorithm.Data_pre_processing import zscore_normalize, regular
from Algorithm.Filters import BandPassFilter, HighPassFilter

class BCG2ECGDataset(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        # load image path and annotations
        self.root = root
        self.signals = os.listdir(root)

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        BCGSignal, ECGsignal = np.load(os.path.join(self.root, self.signals[index]))

        return np.array([zscore_normalize(BCGSignal)]), np.array([zscore_normalize(ECGsignal)])

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.signals)
    
class BCGContrastiveDataset(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        # load image path and annotations
        self.root = root
        self.signals = os.listdir(root)

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)

        ##############################################
        BCGSignal, ECGsignal = np.load(os.path.join(self.root, self.signals[index]))
        BCGSignal = BandPassFilter(signal_=BCGSignal, lowcut=0.5, highcut=25, step=4, fs=100, padlen=50)
        x1 = self._augment_(BCGSignal.copy())  # 同一筆資料做不同增強
        x2 = self._augment_(BCGSignal.copy())
        return zscore_normalize(x1), zscore_normalize(x2)

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.signals)
    
    def _augment_(self, x):
        # 如果 x 是 numpy array，轉為 tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        noise = torch.randn_like(x) * 0.05
        x = x + noise

        return x
    
class BCGSynthesisDataset(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        # load image path and annotations
        self.root = root
        self.signals = os.listdir(root)

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)

        ##############################################
        bcg_signal, bcg_heart_signal = np.load(os.path.join(self.root, self.signals[index]))

        bcg_signal_bp = BandPassFilter(signal_=bcg_signal, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500)


        return np.array([zscore_normalize(bcg_signal_bp)]), np.array([zscore_normalize(bcg_heart_signal)])

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.signals)

class BCGSynthesisDataset_V3(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        # load image path and annotations
        self.root = root
        self.signals = os.listdir(root)

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        file_name = str(self.signals[index])
        conf_label = int(file_name.split('_')[-1].split('.')[0])
        bcg_signal, bcg_heart_signal = np.load(os.path.join(self.root, self.signals[index]))

        bcg_signal_bp = BandPassFilter(signal_=bcg_signal, lowcut=0.5, highcut=25, step=4, fs=100, padlen=500)
        if np.std(bcg_signal_bp) > 0:
            bcg_signal_bp  = zscore_normalize(bcg_signal_bp)
        if np.std(bcg_heart_signal) > 0:
            bcg_heart_signal = zscore_normalize(bcg_heart_signal)

        return np.array([bcg_signal_bp]), np.array([bcg_heart_signal]), np.array([conf_label])

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.signals)
    
if __name__ == '__main__':
    pass
