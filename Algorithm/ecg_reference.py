"""ECG R-peak reference extraction with quality diagnostics.

This module is intentionally separate from the BCG estimator: reference-label
quality must be audited before BCG errors are used for model supervision.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass(frozen=True)
class ECGReferenceResult:
    bpm: float | None
    peaks: np.ndarray
    confidence: float
    valid: bool
    interval_cv: float | None
    reason: str | None = None


def detect_ecg_r_peaks(values, fs=100, min_bpm=35, max_bpm=200):
    """Pan–Tompkins-inspired QRS detection for a 10-second reference ECG."""
    ecg = np.asarray(values, dtype=float).squeeze()
    if ecg.ndim != 1 or len(ecg) < fs * 2 or not np.isfinite(ecg).all() or np.std(ecg) == 0:
        return ECGReferenceResult(None, np.array([], dtype=int), 0.0, False, None, "invalid_signal")
    nyquist = fs / 2
    high = min(20.0, nyquist * 0.9)
    if high <= 5:
        return ECGReferenceResult(None, np.array([], dtype=int), 0.0, False, None, "sampling_rate_too_low")
    sos = butter(3, (5 / nyquist, high / nyquist), btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, ecg)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative ** 2
    window = max(3, int(round(0.12 * fs)))
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")
    minimum_distance = max(1, int(np.floor(fs * 60 / max_bpm)))
    threshold = np.median(integrated) + 1.5 * np.median(np.abs(integrated - np.median(integrated)))
    candidates, properties = find_peaks(integrated, height=threshold, distance=minimum_distance,
                                         prominence=max(np.std(integrated) * 0.15, 1e-12))

    search_radius = max(1, int(round(0.12 * fs)))
    localized = []
    for candidate in candidates:
        left, right = max(0, candidate - search_radius), min(len(filtered), candidate + search_radius + 1)
        localized.append(left + int(np.argmax(np.abs(filtered[left:right]))))
    peaks = np.asarray(sorted(set(localized)), dtype=int)
    if len(peaks) > 1:
        # Localizing two energy peaks may point to the same QRS; retain the stronger one.
        deduplicated = [int(peaks[0])]
        for peak in peaks[1:]:
            if peak - deduplicated[-1] < minimum_distance:
                if abs(filtered[peak]) > abs(filtered[deduplicated[-1]]):
                    deduplicated[-1] = int(peak)
            else:
                deduplicated.append(int(peak))
        peaks = np.asarray(deduplicated, dtype=int)
    if len(peaks) < 3:
        return ECGReferenceResult(None, peaks, 0.0, False, None, "insufficient_qrs_peaks")

    intervals = np.diff(peaks) / fs
    plausible = (intervals >= 60 / max_bpm) & (intervals <= 60 / min_bpm)
    intervals = intervals[plausible]
    if len(intervals) < 2:
        return ECGReferenceResult(None, peaks, 0.0, False, None, "implausible_rr_intervals")
    median_interval = float(np.median(intervals))
    bpm = float(60 / median_interval)
    interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
    interval_quality = float(np.clip(1 - interval_cv / 0.25, 0, 1))
    detection_coverage = float(len(intervals) / max(1, len(peaks) - 1))
    confidence = float(np.sqrt(interval_quality * detection_coverage))
    valid = min_bpm <= bpm <= max_bpm and confidence >= 0.2
    return ECGReferenceResult(bpm if valid else None, peaks, confidence if valid else 0.0,
                              valid, interval_cv, None if valid else "low_quality")
