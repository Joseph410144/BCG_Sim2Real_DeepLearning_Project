"""Heart-rate estimators with a common, research-friendly result interface."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from Algorithm.Filters import BandPassFilter


@dataclass(frozen=True)
class HeartRateResult:
    bpm: float | None
    confidence: float
    method: str
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _as_signal(values):
    signal = np.asarray(values, dtype=float).squeeze()
    if signal.ndim != 1 or len(signal) < 32:
        raise ValueError("BCG input must be a one-dimensional signal with at least 32 samples")
    if not np.isfinite(signal).all() or np.std(signal) == 0:
        raise ValueError("BCG input must contain finite, non-constant values")
    return signal


def _invalid(method, reason, **diagnostics):
    return HeartRateResult(None, 0.0, method, False, {"reason": reason, **diagnostics})


def fft_peak_heart_rate(values, fs=100, min_bpm=42, max_bpm=180):
    """Simple maximum-spectrum-peak baseline."""
    method = "fft_peak"
    signal = _as_signal(values)
    centered = signal - np.mean(signal)
    windowed = centered * np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(windowed), 1 / fs)
    mask = (frequencies >= min_bpm / 60) & (frequencies <= max_bpm / 60)
    candidate_energy = spectrum[mask]
    candidate_freqs = frequencies[mask]
    if not len(candidate_energy) or np.max(candidate_energy) <= 0:
        return _invalid(method, "no_spectral_candidate")
    order = np.argsort(candidate_energy)[::-1]
    best = int(order[0])
    second_energy = candidate_energy[order[1]] if len(order) > 1 else 0.0
    confidence = float(candidate_energy[best] / (candidate_energy[best] + second_energy + 1e-12))
    bpm = float(candidate_freqs[best] * 60)
    return HeartRateResult(bpm, confidence, method, True,
                           {"frequency_hz": float(candidate_freqs[best])})


def autocorrelation_heart_rate(values, fs=100, min_bpm=42, max_bpm=180):
    """Normalized autocorrelation baseline over physiologically plausible lags."""
    method = "autocorrelation"
    signal = _as_signal(values)
    signal = (signal - np.mean(signal)) / np.std(signal)
    correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    correlation /= np.arange(len(signal), 0, -1)
    min_lag = max(1, int(np.floor(fs * 60 / max_bpm)))
    max_lag = min(len(signal) - 1, int(np.ceil(fs * 60 / min_bpm)))
    region = correlation[min_lag:max_lag + 1]
    peaks, _ = find_peaks(region)
    candidates = peaks if len(peaks) else np.arange(len(region))
    if not len(candidates):
        return _invalid(method, "no_autocorrelation_candidate")
    energies = region[candidates]
    order = np.argsort(energies)[::-1]
    best_index = int(candidates[order[0]])
    lag = min_lag + best_index
    best_energy = float(energies[order[0]])
    second_energy = float(energies[order[1]]) if len(order) > 1 else 0.0
    confidence = float(np.clip((best_energy - second_energy) / (abs(best_energy) + 1e-12), 0, 1))
    return HeartRateResult(float(60 * fs / lag), confidence, method, True,
                           {"lag_samples": int(lag), "autocorrelation": best_energy})


def _window_energy(frequencies, spectrum, center, half_width):
    mask = (frequencies >= center - half_width) & (frequencies <= center + half_width)
    return float(np.sum(spectrum[mask]))


def heart_v6(values, fs=100, min_hz=0.7, max_hz=3.0):
    """Clean implementation of the original multi-band harmonic HeartV6 idea.

    It preserves the original V6 signal path while exposing intermediate scores
    and a conservative confidence estimate for subsequent research.
    """
    method = "heart_v6"
    signal = _as_signal(values)
    nyquist = fs / 2
    bands = [(low, high) for low, high in ((1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10))
             if high < nyquist]
    if not bands:
        return _invalid(method, "sampling_rate_too_low")

    fused_spectrum = np.zeros(len(signal), dtype=float)
    for low, high in bands:
        filtered = BandPassFilter(signal, low, high, 4, fs)
        fused_spectrum += np.abs(np.fft.fft(np.abs(filtered)))

    frequencies = np.fft.fftfreq(len(signal), 1 / fs)
    mask = (frequencies > 0) & (frequencies <= max_hz)
    frequencies = frequencies[mask]
    spectrum = fused_spectrum[mask]
    candidate_mask = frequencies > min_hz
    candidate_frequencies = frequencies[candidate_mask]
    if not len(candidate_frequencies):
        return _invalid(method, "no_frequency_candidate")

    harmonic_scores = []
    for frequency in candidate_frequencies:
        harmonic = 1
        score = 0.0
        while harmonic * frequency < max_hz:
            score += _window_energy(frequencies, spectrum, harmonic * frequency, 0.15)
            harmonic += 1
        harmonic_scores.append(score / harmonic)
    harmonic_scores = np.asarray(harmonic_scores)
    if not np.isfinite(harmonic_scores).all() or np.max(harmonic_scores) <= 0:
        return _invalid(method, "no_harmonic_energy")

    order = np.argsort(harmonic_scores)[::-1]
    base_frequency = float(candidate_frequencies[order[0]])
    second_score = float(harmonic_scores[order[1]]) if len(order) > 1 else 0.0
    score_margin = float(np.clip((harmonic_scores[order[0]] - second_score) /
                                 (harmonic_scores[order[0]] + 1e-12), 0, 1))

    harmonic = 1
    selected_frequency = 0.0
    selected_energy = -np.inf
    harmonic_candidates = []
    while harmonic * base_frequency < max_hz:
        frequency = harmonic * base_frequency
        energy = _window_energy(frequencies, spectrum, frequency, 0.15)
        weighted_energy = energy * (2.5 if harmonic == 1 else 1.0)
        harmonic_candidates.append({"frequency_hz": float(frequency), "score": float(weighted_energy)})
        if weighted_energy > selected_energy:
            selected_energy = weighted_energy
            selected_frequency = frequency
        harmonic += 1
    if selected_frequency <= 0.15 or selected_frequency + 0.15 >= nyquist:
        return _invalid(method, "invalid_selected_frequency", selected_frequency_hz=selected_frequency)

    filtered = BandPassFilter(signal, selected_frequency - 0.15, selected_frequency + 0.15, 1, fs)
    filtered = BandPassFilter(filtered, selected_frequency - 0.15, selected_frequency + 0.15, 2, fs)
    peaks, _ = find_peaks(filtered, height=0)
    intervals = np.diff(peaks) / fs
    if not len(intervals) or np.mean(intervals) <= 0:
        return _invalid(method, "insufficient_peaks", peak_count=int(len(peaks)))
    bpm = float(round(60 / np.mean(intervals)))
    interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
    periodicity = float(np.clip(1 - interval_cv, 0, 1))
    confidence = float(np.sqrt(score_margin * periodicity))
    return HeartRateResult(bpm, confidence, method, True, {
        "base_frequency_hz": base_frequency,
        "selected_frequency_hz": float(selected_frequency),
        "peak_count": int(len(peaks)),
        "interval_cv": interval_cv,
        "score_margin": score_margin,
        "harmonic_candidates": harmonic_candidates,
    })


def heart_v7(values, fs=100):
    """Exploratory confidence-gated fusion of HeartV6 and autocorrelation.

    V6 remains the default. Autocorrelation is selected only for two conflict
    patterns associated with harmonic locking in the development dataset. The
    thresholds are explicit so they can be ablated and validated subject-wise.
    """
    method = "heart_v7"
    v6 = heart_v6(values, fs=fs)
    acf = autocorrelation_heart_rate(values, fs=fs)
    if not v6.valid:
        if acf.valid:
            return HeartRateResult(acf.bpm, acf.confidence * 0.5, method, True,
                                   {"selected": "autocorrelation_v6_invalid", "v6": v6.to_dict(),
                                    "autocorrelation": acf.to_dict()})
        return _invalid(method, "both_estimators_invalid", v6=v6.to_dict(), autocorrelation=acf.to_dict())
    selected = v6
    selection_reason = "heart_v6_default"
    if acf.valid and acf.bpm:
        ratio = v6.bpm / acf.bpm
        high_harmonic_conflict = (1.9 <= ratio <= 2.1 and v6.confidence < 0.35 and acf.confidence > 0.8)
        low_harmonic_conflict = (0.6 <= ratio <= 0.8 and v6.confidence < 0.35 and acf.confidence > 0.2)
        if high_harmonic_conflict or low_harmonic_conflict:
            selected = acf
            selection_reason = "autocorrelation_high_conflict" if high_harmonic_conflict else "autocorrelation_low_conflict"
    agreement = 0.0 if not acf.valid or not acf.bpm else float(np.clip(1 - abs(v6.bpm - acf.bpm) / max(v6.bpm, acf.bpm), 0, 1))
    confidence = float(np.clip(0.7 * selected.confidence + 0.3 * agreement, 0, 1))
    return HeartRateResult(selected.bpm, confidence, method, True, {
        "selected": selection_reason,
        "agreement": agreement,
        "heart_v6": v6.to_dict(),
        "autocorrelation": acf.to_dict(),
    })
