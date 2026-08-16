"""Controlled DSP test of low-frequency recovery from amplitude modulation."""

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from Algorithm.Filters import BandPassFilter


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment1_am.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def git_version():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    check=True, capture_output=True, text=True).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN", True


def add_noise(clean, snr_db, rng):
    if snr_db is None:
        return clean.copy()
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / (10 ** (float(snr_db) / 10))
    return clean + rng.normal(0, np.sqrt(noise_power), size=clean.shape)


def low_frequency_metrics(values, fs, target_hz, analysis_band, target_window, tolerance):
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(values), 1 / fs)
    mask = (frequencies >= analysis_band[0]) & (frequencies <= analysis_band[1])
    local_frequencies, local_spectrum = frequencies[mask], spectrum[mask]
    predicted = float(local_frequencies[np.argmax(local_spectrum)])
    target = np.abs(local_frequencies - target_hz) <= target_window
    ratio = float(np.sum(local_spectrum[target]) / (np.sum(local_spectrum) + 1e-12))
    return predicted, ratio, bool(abs(predicted - target_hz) <= tolerance), frequencies, spectrum


def transformed_signals(observed, fs, carrier_hz):
    low = max(0.5, carrier_hz - 2.0)
    high = min(fs / 2 - 0.5, carrier_hz + 2.0)
    filtered = BandPassFilter(observed, low, high, 4, fs)
    return {
        "raw": observed,
        "full_wave_rectified": np.abs(filtered),
        "hilbert_envelope": np.abs(hilbert(filtered)),
        "squared_energy": filtered ** 2,
    }, filtered


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.output_dir / "config.json")
    rng = np.random.default_rng(config["seed"])
    fs = config["sampling_rate_hz"]
    time = np.arange(int(fs * config["duration_seconds"])) / fs
    rows, example = [], None

    for carrier in config["carrier_frequencies_hz"]:
        for modulation in config["modulation_frequencies_hz"]:
            for depth in config["modulation_depths"]:
                clean = (1 + depth * np.cos(2 * np.pi * modulation * time)) \
                    * np.cos(2 * np.pi * carrier * time)
                for snr_db in config["snr_db"]:
                    for replicate in range(config["replicates"]):
                        observed = add_noise(clean, snr_db, rng)
                        methods, filtered = transformed_signals(observed, fs, carrier)
                        spectra = {}
                        for method, representation in methods.items():
                            predicted, ratio, recovered, frequencies, spectrum = low_frequency_metrics(
                                representation, fs, modulation, config["analysis_band_hz"],
                                config["target_window_hz"], config["recovery_tolerance_hz"],
                            )
                            rows.append({
                                "carrier_hz": carrier, "modulation_hz": modulation,
                                "modulation_depth": depth,
                                "snr_db": "inf" if snr_db is None else snr_db,
                                "replicate": replicate, "method": method,
                                "predicted_modulation_hz": predicted,
                                "absolute_frequency_error_hz": abs(predicted - modulation),
                                "target_energy_ratio": ratio, "recovered": int(recovered),
                            })
                            spectra[method] = spectrum
                        if example is None and carrier == 6.0 and modulation == 1.2 \
                                and depth == 0.3 and snr_db == 10 and replicate == 0:
                            example = {"time": time, "clean": clean, "observed": observed,
                                       "bandpassed": filtered, "frequencies_hz": frequencies,
                                       **{f"{key}_representation": value for key, value in methods.items()},
                                       **{f"{key}_spectrum": value for key, value in spectra.items()}}

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    if example:
        np.savez_compressed(args.output_dir / "example_arrays.npz", **example)

    summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "n": len(selected),
            "recovery_rate_percent": float(100 * np.mean([row["recovered"] for row in selected])),
            "mean_absolute_frequency_error_hz": float(np.mean([row["absolute_frequency_error_hz"] for row in selected])),
            "mean_target_energy_ratio": float(np.mean([row["target_energy_ratio"] for row in selected])),
        }
    revision, dirty = git_version()
    result = {
        "statement": ("This synthetic experiment demonstrates that the HeartV6 processing structure can "
                      "recover low-frequency modulation from higher-frequency mechanical components under "
                      "an amplitude-modulation model. It does not establish the mechanism of real BCG."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision, "code_dirty": dirty,
        "seed": config["seed"], "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    methods = list(summary)
    snr_order = ["inf", "20", "10", "0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        means = []
        for snr in snr_order:
            subset = [row for row in rows if row["method"] == method and str(row["snr_db"]) == snr]
            means.append(100 * np.mean([row["recovered"] for row in subset]))
        axes[0].plot(snr_order, means, marker="o", label=method)
    axes[0].set(xlabel="SNR (dB; inf = noise-free)", ylabel="Recovery rate (%)",
                title="Known modulation-frequency recovery")
    axes[0].grid(alpha=0.2); axes[0].legend(fontsize=7)
    frequencies = example["frequencies_hz"]
    for method in methods:
        spectrum = example[f"{method}_spectrum"]
        mask = frequencies <= 10
        axes[1].plot(frequencies[mask], spectrum[mask] / (np.max(spectrum[mask]) + 1e-12),
                     label=method)
    axes[1].axvline(1.2, color="black", linestyle="--", linewidth=1, label="known modulation")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Normalized magnitude",
                title="Representative AM example")
    axes[1].grid(alpha=0.2); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "am_recovery.png", dpi=180)
    fig.savefig(args.output_dir / "am_recovery.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
