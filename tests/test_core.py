import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Algorithm.Bcg_signal_synthesis_function import BedBCGGenerator
from Algorithm.Filters import BandPassFilter
from Dataset.BCG_Dataset import BCGSynthesisDataset, RealBCGHeartRateDataset
from Model.LSTM import LSTM_BCGFilter_Pre
from Model.Loss_Function import MorletCWTLoss
from bcg_signal_synthesize import split_counts
from validate_performance import metrics
from Algorithm.heart_rate import autocorrelation_heart_rate, fft_peak_heart_rate, heart_v6, heart_v7
from Dataset.metadata import parse_real_recording_name
from Algorithm.ecg_reference import detect_ecg_r_peaks
from Dataset.splits import leave_one_subject_out_folds
from analyze_failures import classify_error


class CoreTests(unittest.TestCase):
    def test_split_counts_preserves_total(self):
        self.assertEqual(split_counts(101, (0.8, 0.1, 0.1)), (80, 10, 11))
        with self.assertRaises(ValueError):
            split_counts(10, (0.8, 0.2, 0.2))

    def test_filter_preserves_shape_and_finite_values(self):
        x = np.sin(np.linspace(0, 20, 1000))
        result = BandPassFilter(x, 0.5, 25, 4, 100, padlen=100)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_dataset_filters_and_sorts_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "b.npy", np.ones((2, 1000), dtype=np.float32))
            np.save(root / "a.npy", np.ones((2, 1000), dtype=np.float32))
            (root / ".DS_Store").write_text("ignored")
            dataset = BCGSynthesisDataset(directory)
            self.assertEqual(dataset.signals, ["a.npy", "b.npy"])
            signal, target = dataset[0]
            self.assertEqual(signal.shape, (1, 1000))
            self.assertEqual(target.shape, (1, 1000))

    def test_real_dataset_subject_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            time = np.arange(1000) / 100
            ecg = np.zeros(1000)
            for peak in range(50, 1000, 75):
                ecg[peak] = 10
            bcg = np.sin(2 * np.pi * 80 / 60 * time)
            np.save(root / "2025-01-01_1200_28cm_Subject_2_heart_0.npy", [bcg, ecg])
            np.save(root / "2025-01-01_1200_28cm_Subject_3_heart_0.npy", [bcg, ecg])
            dataset = RealBCGHeartRateDataset(directory, subject_ids=[2])
            signal, bpm, subject, filename = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(signal.shape, (1, 1000))
            self.assertAlmostEqual(float(bpm), 80, delta=1)
            self.assertEqual(subject, 2)

    def test_model_and_loss_backward(self):
        model = LSTM_BCGFilter_Pre(64, 1, 4, 1, 0, 1, True, 1)
        x = torch.randn(2, 1, 64)
        target = torch.randn(2, 1, 64)
        output = model(x)
        self.assertEqual(output.shape, target.shape)
        loss = MorletCWTLoss(fs=100, fmin=1, fmax=10, num_freqs=4, kernel_size=31)(output, target)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_generator_output(self):
        noisy, clean = BedBCGGenerator(fs=100).generate(duration=1)
        self.assertEqual(np.shape(noisy), (100,))
        self.assertEqual(np.shape(clean), (100,))
        self.assertTrue(np.isfinite(noisy).all())

    def test_metrics(self):
        result = metrics([60, 70, 80], [61, 69, 82])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mae_bpm"], 4 / 3)
        self.assertEqual(result["within_3_bpm_percent"], 100.0)

    def test_heart_rate_estimators_on_periodic_signal(self):
        fs = 100
        time = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 1.2 * time) + 0.4 * np.sin(2 * np.pi * 2.4 * time)
        self.assertAlmostEqual(fft_peak_heart_rate(signal, fs).bpm, 72, delta=1)
        self.assertAlmostEqual(autocorrelation_heart_rate(signal, fs).bpm, 72, delta=2)
        self.assertAlmostEqual(heart_v6(signal, fs).bpm, 72, delta=3)
        self.assertAlmostEqual(heart_v7(signal, fs).bpm, 72, delta=3)

    def test_real_filename_metadata(self):
        metadata = parse_real_recording_name("2025-08-22_1146_47cm_Subject_10_heart_1.npy")
        self.assertEqual(metadata.subject_id, 10)
        self.assertEqual(metadata.distance_cm, 47)

    def test_ecg_reference_detector(self):
        fs = 100
        ecg = np.zeros(1000)
        expected_peaks = np.arange(50, 1000, 75)
        for peak in expected_peaks:
            indices = np.arange(max(0, peak - 4), min(len(ecg), peak + 5))
            ecg[indices] += np.exp(-0.5 * ((indices - peak) / 1.5) ** 2)
        result = detect_ecg_r_peaks(ecg, fs)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.bpm, 80, delta=1)

    def test_subject_folds_do_not_leak(self):
        folds = leave_one_subject_out_folds(range(1, 11))
        self.assertEqual(len(folds), 10)
        for fold in folds:
            train, val, test = set(fold.train_subjects), set(fold.val_subjects), set(fold.test_subjects)
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(range(1, 11)))

    def test_failure_classification(self):
        self.assertEqual(classify_error(70, 72), "within_5_bpm")
        self.assertEqual(classify_error(70, 140), "double_rate")
        self.assertEqual(classify_error(80, 40), "half_rate")


if __name__ == "__main__":
    unittest.main()
