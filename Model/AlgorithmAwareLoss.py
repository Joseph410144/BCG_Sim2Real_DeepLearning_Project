"""Differentiable objectives connecting a BCG filter to downstream HR estimation."""

import torch
from torch import nn
import torch.nn.functional as F
from scipy import signal as scipy_signal


class HarmonicHeartRateLoss(nn.Module):
    """Maximize rectified-envelope energy at ECG-derived HR and its harmonics."""

    def __init__(self, fs=100, min_hz=0.7, max_hz=3.0, bandwidth_hz=0.12,
                 harmonics=3, eps=1e-8):
        super().__init__()
        self.fs = fs
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.bandwidth_hz = bandwidth_hz
        self.harmonics = harmonics
        self.eps = eps

    def forward(self, signal, target_bpm):
        if signal.ndim == 3:
            signal = signal.squeeze(1)
        target_hz = target_bpm.reshape(-1, 1) / 60.0
        envelope = torch.abs(signal - signal.mean(dim=-1, keepdim=True))
        window = torch.hann_window(signal.shape[-1], device=signal.device, dtype=signal.dtype)
        spectrum = torch.abs(torch.fft.rfft(envelope * window, dim=-1)) ** 2
        frequencies = torch.fft.rfftfreq(signal.shape[-1], 1 / self.fs).to(signal.device)
        valid_band = ((frequencies >= self.min_hz) & (frequencies <= self.max_hz)).to(signal.dtype)
        total_energy = torch.sum(spectrum * valid_band, dim=-1) + self.eps

        target_weight = torch.zeros_like(spectrum)
        for harmonic in range(1, self.harmonics + 1):
            centers = harmonic * target_hz
            target_weight = target_weight + torch.exp(
                -0.5 * ((frequencies.reshape(1, -1) - centers) / self.bandwidth_hz) ** 2
            ) / harmonic
        target_weight = target_weight * valid_band
        target_energy = torch.sum(spectrum * target_weight, dim=-1)
        ratio = target_energy / (total_energy + self.eps)
        return -torch.log(ratio + self.eps).mean()


class HeartV6SurrogateLoss(nn.Module):
    """Differentiable approximation of HeartV6's candidate-scoring path.

    The original estimator applies six overlapping carrier-band filters,
    rectifies each output, fuses their spectra, and scores fundamental-HR
    candidates using energy near their harmonics.  Ideal FFT masks replace the
    non-differentiable SciPy IIR filters, while all remaining operations mirror
    that decision path closely.
    """

    def __init__(self, fs=100, min_hz=0.7, max_hz=3.0,
                 bandwidth_hz=0.15, target_sigma_hz=0.08,
                 temperature=0.25, margin=0.15, eps=1e-8):
        super().__init__()
        self.fs = fs
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.bandwidth_hz = bandwidth_hz
        self.target_sigma_hz = target_sigma_hz
        self.temperature = temperature
        self.margin = margin
        self.eps = eps
        self.bands = ((1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10))
        self._response_cache = {}

    def _filter_responses(self, n_samples, device, dtype):
        key = (n_samples, str(device), dtype)
        if key not in self._response_cache:
            frequencies = torch.fft.rfftfreq(n_samples, 1 / self.fs).cpu().numpy()
            responses = []
            for low_hz, high_hz in self.bands:
                numerator, denominator = scipy_signal.butter(
                    4, [low_hz, high_hz], btype="bandpass", fs=self.fs
                )
                _, response = scipy_signal.freqz(
                    numerator, denominator, worN=frequencies, fs=self.fs
                )
                # scipy filtfilt applies the filter forward and backward.
                responses.append(torch.as_tensor(abs(response) ** 2,
                                                 device=device, dtype=dtype))
            self._response_cache[key] = responses
        return self._response_cache[key]

    def candidate_scores(self, signal):
        if signal.ndim == 3:
            signal = signal.squeeze(1)
        n_samples = signal.shape[-1]
        frequencies = torch.fft.rfftfreq(
            n_samples, 1 / self.fs, device=signal.device
        ).to(signal.dtype)
        source_spectrum = torch.fft.rfft(signal, dim=-1)
        fused = torch.zeros_like(frequencies).expand(signal.shape[0], -1).clone()
        for response in self._filter_responses(n_samples, signal.device, signal.dtype):
            band_signal = torch.fft.irfft(source_spectrum * response, n=n_samples, dim=-1)
            fused = fused + torch.abs(torch.fft.rfft(torch.abs(band_signal), dim=-1))

        candidate_mask = (frequencies > self.min_hz) & (frequencies <= self.max_hz)
        candidates = frequencies[candidate_mask]
        scores = []
        for candidate in candidates:
            harmonic_scores = []
            harmonic = 1
            while harmonic * float(candidate) < self.max_hz:
                center = harmonic * candidate
                window = (torch.abs(frequencies - center) <= self.bandwidth_hz).to(signal.dtype)
                harmonic_scores.append(torch.sum(fused * window, dim=-1))
                harmonic += 1
            if harmonic_scores:
                # HeartV6 divides by the post-loop harmonic counter (N + 1).
                scores.append(torch.stack(harmonic_scores).sum(dim=0) /
                              (len(harmonic_scores) + 1))
            else:
                scores.append(torch.zeros(signal.shape[0], device=signal.device,
                                          dtype=signal.dtype))
        return candidates, torch.stack(scores, dim=-1)

    def forward(self, signal, target_bpm):
        candidates, scores = self.candidate_scores(signal)
        target_hz = target_bpm.reshape(-1, 1) / 60.0
        target_distribution = torch.exp(
            -0.5 * ((candidates.reshape(1, -1) - target_hz) / self.target_sigma_hz) ** 2
        )
        target_distribution = target_distribution / (
            target_distribution.sum(dim=-1, keepdim=True) + self.eps
        )
        logits = torch.log(scores + self.eps) / self.temperature
        classification = torch.sum(
            -target_distribution * F.log_softmax(logits, dim=-1), dim=-1
        )

        target_score = torch.sum(scores * target_distribution, dim=-1)
        distractor_mask = torch.abs(candidates.reshape(1, -1) - target_hz) > 0.2
        distractor = torch.where(
            distractor_mask, scores, torch.full_like(scores, -torch.inf)
        ).max(dim=-1).values
        ranking = F.relu(self.margin - torch.log(target_score + self.eps)
                         + torch.log(distractor + self.eps))
        return (classification + ranking).mean()


class SpectralReconstructionLoss(nn.Module):
    def __init__(self, fft_sizes=(64, 256, 512), eps=1e-7):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def forward(self, prediction, target):
        prediction, target = prediction.squeeze(1), target.squeeze(1)
        total = prediction.new_tensor(0.0)
        for fft_size in self.fft_sizes:
            window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
            pred = torch.abs(torch.stft(prediction, fft_size, fft_size // 4, fft_size,
                                        window, return_complex=True)) + self.eps
            truth = torch.abs(torch.stft(target, fft_size, fft_size // 4, fft_size,
                                         window, return_complex=True)) + self.eps
            convergence = torch.linalg.vector_norm(truth - pred) / (torch.linalg.vector_norm(truth) + self.eps)
            log_magnitude = F.l1_loss(torch.log(pred), torch.log(truth))
            total = total + convergence + log_magnitude
        return total / len(self.fft_sizes)


def synthetic_filter_loss(output, target, spectral_loss, time_weight=1.0, spectral_weight=0.5):
    time_loss = F.mse_loss(output, target)
    frequency_loss = spectral_loss(output, target)
    return time_weight * time_loss + spectral_weight * frequency_loss, {
        "time": time_loss.detach(), "spectral": frequency_loss.detach()
    }


def real_algorithm_aware_loss(output, input_signal, target_bpm, harmonic_loss,
                              hr_weight=1.0, identity_weight=0.05, residual_tv_weight=0.01):
    hr_loss = harmonic_loss(output, target_bpm)
    identity_loss = F.l1_loss(output, input_signal)
    residual = output - input_signal
    total_variation = torch.mean(torch.abs(residual[..., 1:] - residual[..., :-1]))
    total = hr_weight * hr_loss + identity_weight * identity_loss + residual_tv_weight * total_variation
    return total, {"hr": hr_loss.detach(), "identity": identity_loss.detach(),
                   "residual_tv": total_variation.detach()}
