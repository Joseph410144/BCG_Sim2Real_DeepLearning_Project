"""Read-only instrumentation for the production HeartV6 signal path."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from Algorithm.Filters import BandPassFilter
from Algorithm.heart_rate import HeartRateResult, heart_v6


HEART_V6_BANDS = ((1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10))


@dataclass(frozen=True)
class HeartV6Trace:
    result: HeartRateResult
    fs: int
    raw_signal: np.ndarray
    bands_hz: np.ndarray
    band_signals: np.ndarray
    rectified_bands: np.ndarray
    frequencies_hz: np.ndarray
    band_spectra: np.ndarray
    fused_spectrum: np.ndarray
    candidate_frequencies_hz: np.ndarray
    harmonic_scores: np.ndarray
    selected_frequency_hz: float
    narrowband_signal: np.ndarray
    peaks: np.ndarray
    intervals_seconds: np.ndarray

    def array_dict(self):
        return {
            "raw_signal": self.raw_signal,
            "bands_hz": self.bands_hz,
            "band_signals": self.band_signals,
            "rectified_bands": self.rectified_bands,
            "frequencies_hz": self.frequencies_hz,
            "band_spectra": self.band_spectra,
            "fused_spectrum": self.fused_spectrum,
            "candidate_frequencies_hz": self.candidate_frequencies_hz,
            "harmonic_scores": self.harmonic_scores,
            "narrowband_signal": self.narrowband_signal,
            "peaks": self.peaks,
            "intervals_seconds": self.intervals_seconds,
        }

    def metadata_dict(self):
        return {
            "fs": self.fs,
            "selected_frequency_hz": self.selected_frequency_hz,
            "heart_rate_result": self.result.to_dict(),
            "bands_hz": self.bands_hz.tolist(),
        }


def _window_energy(frequencies, spectrum, center, half_width=0.15):
    mask = (frequencies >= center - half_width) & (frequencies <= center + half_width)
    return float(np.sum(spectrum[mask]))


def trace_heart_v6(values, fs=100, min_hz=0.7, max_hz=3.0):
    """Capture intermediates while leaving :func:`heart_v6` unchanged.

    The production estimator is called first and remains the authority for the
    reported result. Instrumentation independently reconstructs its documented
    signal path and raises if the final peak-derived BPM disagrees.
    """
    raw = np.asarray(values, dtype=float).squeeze()
    if raw.ndim != 1 or len(raw) < 32 or not np.isfinite(raw).all() or np.std(raw) == 0:
        raise ValueError("BCG input must be finite, non-constant, and one-dimensional")
    result = heart_v6(raw, fs=fs, min_hz=min_hz, max_hz=max_hz)
    if not result.valid:
        raise ValueError(f"HeartV6 trace requires a valid result: {result.diagnostics}")

    band_signals = np.stack([
        BandPassFilter(raw, low, high, 4, fs) for low, high in HEART_V6_BANDS
        if high < fs / 2
    ])
    rectified = np.abs(band_signals)
    full_spectra = np.abs(np.fft.fft(rectified, axis=-1))
    full_frequencies = np.fft.fftfreq(len(raw), 1 / fs)
    positive = (full_frequencies > 0) & (full_frequencies <= 10)
    frequencies = full_frequencies[positive]
    band_spectra = full_spectra[:, positive]
    fused = np.sum(band_spectra, axis=0)

    scoring_mask = frequencies <= max_hz
    scoring_frequencies = frequencies[scoring_mask]
    scoring_spectrum = fused[scoring_mask]
    candidate_mask = scoring_frequencies > min_hz
    candidates = scoring_frequencies[candidate_mask]
    scores = []
    for frequency in candidates:
        harmonic = 1
        score = 0.0
        while harmonic * frequency < max_hz:
            score += _window_energy(scoring_frequencies, scoring_spectrum, harmonic * frequency)
            harmonic += 1
        scores.append(score / harmonic)
    scores = np.asarray(scores)

    selected = float(result.diagnostics["selected_frequency_hz"])
    narrow = BandPassFilter(raw, selected - 0.15, selected + 0.15, 1, fs)
    narrow = BandPassFilter(narrow, selected - 0.15, selected + 0.15, 2, fs)
    peaks, _ = find_peaks(narrow, height=0)
    intervals = np.diff(peaks) / fs
    traced_bpm = float(round(60 / np.mean(intervals)))
    if traced_bpm != result.bpm:
        raise RuntimeError(f"Trace diverged from production HeartV6: {traced_bpm} != {result.bpm}")

    return HeartV6Trace(
        result=result, fs=fs, raw_signal=raw,
        bands_hz=np.asarray(HEART_V6_BANDS, dtype=float),
        band_signals=band_signals, rectified_bands=rectified,
        frequencies_hz=frequencies, band_spectra=band_spectra,
        fused_spectrum=fused, candidate_frequencies_hz=candidates,
        harmonic_scores=scores, selected_frequency_hz=selected,
        narrowband_signal=narrow, peaks=peaks.astype(int), intervals_seconds=intervals,
    )
