"""Select non-cherry-picked real cases and export complete HeartV6 traces."""

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
from scipy.signal import spectrogram

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.Filters import BandPassFilter
from Algorithm.heart_v6_trace import trace_heart_v6


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT.parent / "Dataset" / "BCG" / "DeepLearningData" / "BCG_ECG_10sec"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "experiment1_real_trace.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "baselines" / "per_sample.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
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


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify(error, definitions):
    for label, (lower, upper) in definitions.items():
        if error >= lower and (upper is None or error <= upper):
            return label
    return None


def select_cases(baseline_rows, manifest_rows, config):
    manifest = {row["source_filename"]: row for row in manifest_rows}
    candidates = []
    for row in baseline_rows:
        if row["filename"] not in manifest:
            continue
        error = abs(float(row["heart_v6_hr"]) - float(row["ecg_hr"]))
        label = classify(error, config["error_classes_bpm"])
        if label:
            candidates.append({**row, **manifest[row["filename"]],
                               "absolute_error_bpm": error, "selection_class": label})
    rng = np.random.default_rng(config["seed"])
    selected, used_datasets, used_sources = [], set(), set()
    for label in config["error_classes_bpm"]:
        pool = [row for row in candidates if row["selection_class"] == label]
        order = rng.permutation(len(pool))
        pool = [pool[index] for index in order]
        chosen = []
        for prefer_new_dataset, prefer_new_source in ((True, True), (False, True), (False, False)):
            for row in pool:
                if row in chosen:
                    continue
                dataset_id = int(row["legacy_dataset_id"])
                source_id = row["continuous_source_id"]
                if prefer_new_dataset and dataset_id in used_datasets:
                    continue
                if prefer_new_source and source_id in used_sources:
                    continue
                chosen.append(row); used_datasets.add(dataset_id); used_sources.add(source_id)
                if len(chosen) == config["cases_per_error_class"]:
                    break
            if len(chosen) == config["cases_per_error_class"]:
                break
        if len(chosen) != config["cases_per_error_class"]:
            raise ValueError(f"Could not select enough cases for {label}")
        selected.extend(chosen)
    return selected


def plot_trace(trace, ecg_hr, title, path):
    fs = trace.fs
    time = np.arange(len(trace.raw_signal)) / fs
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    axes = axes.ravel()
    axes[0].plot(time, trace.raw_signal, linewidth=0.8)
    axes[0].set(title="Preprocessed BCG", xlabel="Time (s)", ylabel="z-score")

    f, t, power = spectrogram(trace.raw_signal, fs=fs, nperseg=256, noverlap=224)
    mask = f <= 12
    axes[1].pcolormesh(t, f[mask], 10 * np.log10(power[mask] + 1e-12), shading="auto")
    axes[1].set(title="BCG spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")

    for index, ((low, high), values) in enumerate(zip(trace.bands_hz, trace.band_signals)):
        axes[2].plot(time, values / (np.std(values) + 1e-12) + index * 4,
                     linewidth=0.65, label=f"{low:g}-{high:g} Hz")
    axes[2].set(title="Mechanical-frequency band outputs", xlabel="Time (s)", yticks=[])
    axes[2].legend(fontsize=7, ncol=2)

    for index, ((low, high), values) in enumerate(zip(trace.bands_hz, trace.rectified_bands)):
        normalized = values / (np.std(values) + 1e-12)
        axes[3].plot(time, normalized + index * 4, linewidth=0.65,
                     label=f"{low:g}-{high:g} Hz")
    axes[3].set(title="Full-wave rectified band representations", xlabel="Time (s)", yticks=[])

    for (low, high), values in zip(trace.bands_hz, trace.band_spectra):
        axes[4].plot(trace.frequencies_hz, values / (np.max(values) + 1e-12),
                     alpha=0.8, label=f"{low:g}-{high:g} Hz")
    axes[4].set_xlim(0, 10); axes[4].set(title="Individual rectified-band spectra",
                                        xlabel="Frequency (Hz)", ylabel="Normalized magnitude")
    axes[4].legend(fontsize=7, ncol=2)

    axes[5].plot(trace.frequencies_hz, trace.fused_spectrum)
    axes[5].axvline(ecg_hr / 60, color="green", linestyle="--", label="ECG HR")
    axes[5].axvline(trace.selected_frequency_hz, color="red", linestyle=":", label="selected")
    axes[5].set_xlim(0, 3); axes[5].set(title="Fused rectified spectrum",
                                       xlabel="Frequency (Hz)", ylabel="Magnitude")
    axes[5].legend(fontsize=8)

    axes[6].plot(trace.candidate_frequencies_hz, trace.harmonic_scores)
    axes[6].axvline(trace.result.diagnostics["base_frequency_hz"], color="purple",
                    linestyle="--", label="base candidate")
    axes[6].set(title="Harmonic candidate scores", xlabel="Candidate fundamental (Hz)",
                ylabel="Score"); axes[6].legend(fontsize=8)

    axes[7].plot(time, trace.narrowband_signal, linewidth=0.8)
    axes[7].plot(trace.peaks / fs, trace.narrowband_signal[trace.peaks], "rx", markersize=5)
    axes[7].set(title=f"Narrowband refinement and peaks (HR={trace.result.bpm:.1f} BPM)",
                xlabel="Time (s)", ylabel="Amplitude")
    for axis in axes:
        axis.grid(alpha=0.15)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.output_dir / "config.json")
    selected = select_cases(read_csv(args.baseline_csv), read_csv(args.manifest), config)
    selection_fields = [
        "selection_class", "source_filename", "legacy_dataset_id", "recording_date",
        "recording_time", "sensor_distance_cm", "recording_session_id",
        "continuous_source_id", "allowed_split_group", "ecg_hr", "heart_v6_hr",
        "absolute_error_bpm", "heart_v6_confidence",
    ]
    with (args.output_dir / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)

    summaries = []
    for index, row in enumerate(selected):
        pair = np.load(args.data_dir / row["source_filename"])
        bcg = zscore_normalize(BandPassFilter(pair[0], 0.5, 25, 4, 100, padlen=500))
        trace = trace_heart_v6(bcg, fs=config["sampling_rate_hz"])
        case_id = f"case_{index + 1:02d}_{row['selection_class']}"
        np.savez_compressed(args.output_dir / f"{case_id}_arrays.npz", **trace.array_dict())
        metadata = {**trace.metadata_dict(), **{key: row[key] for key in selection_fields}}
        (args.output_dir / f"{case_id}_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        title = (f"{row['selection_class']} | legacy dataset {row['legacy_dataset_id']} | "
                 f"ECG {float(row['ecg_hr']):.1f}, HeartV6 {trace.result.bpm:.1f} BPM")
        plot_trace(trace, float(row["ecg_hr"]), title, args.output_dir / case_id)
        summaries.append({"case_id": case_id, **metadata})

    revision, dirty = git_version()
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision, "code_dirty": dirty,
        "seed": config["seed"], "selection_rule": config["selection_rule"],
        "input_manifest": str(args.manifest.resolve()),
        "baseline_csv": str(args.baseline_csv.resolve()),
        "case_count": len(selected), "recording_session_limitation": (
            "Continuous source ancestry is verified; higher recording-session grouping remains UNKNOWN."
        ),
        "cases": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in run.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
