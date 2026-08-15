"""Visualize raw BCG, model-filtered BCG, and reference ECG for one recording."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.Filters import BandPassFilter
from Model.LSTM import LSTM_BCGFilter_Pre
from validate_performance import DEFAULT_DATA, DEFAULT_WEIGHTS, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--file", type=Path, help="Specific .npy file; defaults to the first sorted file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "signal_example.png")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def normalized_spectrum(signal, fs=100):
    magnitude = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), 1 / fs)
    return frequencies, zscore_normalize(magnitude)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.data_dir.glob("*.npy"))
    path = args.file or (files[0] if files else None)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"No input .npy file found: {path or args.data_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    pair = np.load(path)
    if pair.shape != (2, 1000):
        raise ValueError(f"Expected (2, 1000), got {pair.shape}: {path}")

    bcg = zscore_normalize(BandPassFilter(pair[0], 0.5, 25, 4, 100, padlen=500))
    ecg = zscore_normalize(BandPassFilter(pair[1], 0.5, 25, 4, 100, padlen=500))
    device = resolve_device(args.device)
    model = LSTM_BCGFilter_Pre(1000, 1, 128, 1, 0.2, 6, True, 1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        filtered = model(torch.as_tensor(bcg, dtype=torch.float32, device=device)[None, None]).squeeze().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)
    series = ((bcg, "Raw real-world BCG", "#1f77b4"),
              (filtered, "Model-filtered BCG", "#d62728"),
              (ecg, "Reference ECG", "black"))
    for row, (values, title, color) in enumerate(series):
        axes[row, 0].plot(values, color=color, linewidth=1)
        axes[row, 0].set_title(f"{title} — time")
        axes[row, 0].set_xlim(0, len(values))
        frequencies, spectrum = normalized_spectrum(values)
        axes[row, 1].plot(frequencies, spectrum, color=color, linewidth=1)
        axes[row, 1].set_title(f"{title} — frequency")
        axes[row, 1].set_xlim(0, 10)
        for axis in axes[row]:
            axis.grid(True, linestyle=":", alpha=0.5)
    axes[2, 0].set_xlabel("Samples (100 Hz)")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(path.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
