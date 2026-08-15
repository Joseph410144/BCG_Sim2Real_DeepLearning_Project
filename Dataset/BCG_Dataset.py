import os
import torch
import numpy as np

from scipy import signal
from torch.utils.data import Dataset
from scipy.fft import fft, ifft, fftfreq
from Algorithm.Data_pre_processing import zscore_normalize, regular
from Algorithm.Filters import BandPassFilter, HighPassFilter
from Algorithm.ecg_reference import detect_ecg_r_peaks
from Dataset.metadata import parse_real_recording_name


def _npy_files(root):
    """Return a deterministic list of dataset files and fail with a useful error."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files = sorted(name for name in os.listdir(root) if name.lower().endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in dataset directory: {root}")
    return files

class BCG2ECGDataset(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        # load image path and annotations
        self.root = root
        self.signals = _npy_files(root)

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
        self.signals = _npy_files(root)

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
        self.signals = _npy_files(root)

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
        self.signals = _npy_files(root)

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


class RealBCGHeartRateDataset(Dataset):
    """Real BCG input with ECG-derived HR supervision and subject filtering."""

    def __init__(self, root, subject_ids=None, fs=100):
        self.root = root
        self.fs = fs
        allowed = None if subject_ids is None else {int(subject) for subject in subject_ids}
        self.signals = []
        for filename in _npy_files(root):
            metadata = parse_real_recording_name(filename)
            if allowed is None or metadata.subject_id in allowed:
                self.signals.append(filename)
        if not self.signals:
            raise FileNotFoundError(f"No recordings matched requested subjects in {root}")
        self._hr_cache = {}

    def __getitem__(self, index):
        filename = self.signals[index]
        pair = np.load(os.path.join(self.root, filename))
        if pair.shape != (2, 1000) or not np.isfinite(pair).all():
            raise ValueError(f"Expected finite (2, 1000) recording: {filename}")
        bcg = BandPassFilter(pair[0], 0.5, 25, 4, self.fs, padlen=500)
        bcg = zscore_normalize(bcg).astype(np.float32)
        if filename not in self._hr_cache:
            reference = detect_ecg_r_peaks(pair[1], fs=self.fs)
            if not reference.valid:
                raise ValueError(f"Invalid ECG reference for {filename}: {reference.reason}")
            self._hr_cache[filename] = np.float32(reference.bpm)
        metadata = parse_real_recording_name(filename)
        return np.asarray([bcg], dtype=np.float32), self._hr_cache[filename], metadata.subject_id, filename

    def __len__(self):
        return len(self.signals)

if __name__ == '__main__':
    pass
