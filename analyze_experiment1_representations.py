"""Quantitative, read-only analysis of Experiment 1 synthetic and real traces."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_controlled_am_experiment import add_noise, transformed_signals


METHODS = ("raw", "full_wave_rectified", "hilbert_envelope", "squared_energy")


def spectral_metrics(values, fs, target_hz, band=(0.5, 3.0), half_width=0.1):
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(centered), 1 / fs)
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    f, s = frequencies[mask], spectrum[mask] ** 2
    target_mask = np.abs(f - target_hz) <= half_width
    target_energy = float(np.sum(s[target_mask]))
    distractor_centers = f[np.abs(f - target_hz) > 2 * half_width]
    distractor = max((float(np.sum(s[np.abs(f - center) <= half_width]))
                      for center in distractor_centers), default=0.0)
    predicted = float(f[np.argmax(s)])
    return {
        "predicted_hz": predicted,
        "absolute_error_hz": abs(predicted - target_hz),
        "recovered": int(abs(predicted - target_hz) <= 0.15),
        "target_energy": target_energy,
        "strongest_distractor_energy": distractor,
        "target_to_distractor_ratio": target_energy / (distractor + 1e-12),
        "frequencies": f,
        "spectrum": s,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_grid(rows, method, row_key, row_values, col_key, col_values, metric):
    grid = np.full((len(row_values), len(col_values)), np.nan)
    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            selected = [float(r[metric]) for r in rows if r["method"] == method
                        and r[row_key] == rv and r[col_key] == cv]
            grid[i, j] = np.mean(selected)
    return grid


def plot_heatmaps(rows, config, output):
    depths = config["modulation_depths"]
    snrs = ["inf" if value is None else str(value) for value in config["snr_db"]]
    carriers = config["carrier_frequencies_hz"]
    modulations = config["modulation_frequencies_hz"]
    for metric, label, scale in (("recovered", "Recovery rate (%)", 100),
                                  ("target_to_distractor_ratio", "Target/distractor ratio", 1)):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for column, method in enumerate(METHODS):
            grids = (
                mean_grid(rows, method, "modulation_depth", depths, "snr_db", snrs, metric) * scale,
                mean_grid(rows, method, "carrier_hz", carriers, "modulation_hz", modulations, metric) * scale,
            )
            for row_index, (grid, xlabels, ylabels, xname, yname) in enumerate((
                    (grids[0], snrs, depths, "SNR (dB)", "Modulation depth"),
                    (grids[1], modulations, carriers, "Modulation frequency (Hz)", "Carrier (Hz)"))):
                axis = axes[row_index, column]
                image = axis.imshow(grid, aspect="auto", cmap="viridis")
                axis.set(xticks=range(len(xlabels)), xticklabels=xlabels,
                         yticks=range(len(ylabels)), yticklabels=ylabels,
                         xlabel=xname, ylabel=yname, title=method)
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        axis.text(j, i, f"{grid[i,j]:.1f}" if scale == 100 else f"{grid[i,j]:.2f}",
                                  ha="center", va="center", color="white", fontsize=7)
                fig.colorbar(image, ax=axis, shrink=0.72)
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(output / f"am_{metric}_heatmaps.png", dpi=180)
        plt.close(fig)


def plot_method_comparison(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (metric, title) in enumerate((("absolute_error_hz", "Frequency error (Hz)"),
                                         ("target_to_distractor_ratio", "Target/distractor ratio"),
                                         ("recovered", "Recovery rate"))):
        values = [[float(row[metric]) for row in rows if row["method"] == method]
                  for method in METHODS]
        if metric == "recovered":
            axes[i].bar(range(4), [100 * np.mean(v) for v in values])
            axes[i].set_ylabel("Percent")
        else:
            axes[i].boxplot(values, tick_labels=[m.replace("_", "\n") for m in METHODS], showfliers=False)
        axes[i].set_title(title)
        axes[i].grid(alpha=0.2, axis="y")
    axes[2].set_xticks(range(4), [m.replace("_", "\n") for m in METHODS])
    fig.tight_layout()
    fig.savefig(output / "am_method_comparison.png", dpi=180)
    plt.close(fig)


def plot_am_case(case, output, label):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(case["time"], case["observed"], linewidth=0.7, label="observed")
    axes[0].plot(case["time"], case["clean"], linewidth=0.7, alpha=0.65, label="clean")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title=label)
    axes[0].legend()
    for method in METHODS:
        result = case["metrics"][method]
        axes[1].plot(result["frequencies"], result["spectrum"] / (np.max(result["spectrum"]) + 1e-12),
                     label=f"{method}: err={result['absolute_error_hz']:.2f}, T/D={result['target_to_distractor_ratio']:.2f}")
    axes[1].axvline(case["modulation_hz"], color="black", linestyle="--", label="known modulation")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Normalized magnitude", xlim=(0.5, 3.0))
    axes[1].legend(fontsize=7)
    for axis in axes: axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_am(config, output):
    rng = np.random.default_rng(config["seed"])
    fs = config["sampling_rate_hz"]
    time = np.arange(int(fs * config["duration_seconds"])) / fs
    rows, cases = [], []
    for carrier in config["carrier_frequencies_hz"]:
        for modulation in config["modulation_frequencies_hz"]:
            for depth in config["modulation_depths"]:
                clean = (1 + depth * np.cos(2*np.pi*modulation*time)) * np.cos(2*np.pi*carrier*time)
                for snr_value in config["snr_db"]:
                    for replicate in range(config["replicates"]):
                        observed = add_noise(clean, snr_value, rng)
                        signals, _ = transformed_signals(observed, fs, carrier)
                        metrics = {method: spectral_metrics(signal, fs, modulation)
                                   for method, signal in signals.items()}
                        case = {"carrier_hz": carrier, "modulation_hz": modulation,
                                "modulation_depth": depth,
                                "snr_db": "inf" if snr_value is None else str(snr_value),
                                "replicate": replicate, "time": time, "clean": clean,
                                "observed": observed, "metrics": metrics}
                        cases.append(case)
                        for method, result in metrics.items():
                            rows.append({"carrier_hz": carrier, "modulation_hz": modulation,
                                         "modulation_depth": depth, "snr_db": case["snr_db"],
                                         "replicate": replicate, "method": method,
                                         **{key: value for key, value in result.items()
                                            if key not in ("frequencies", "spectrum")}})
    write_csv(output / "am_quantitative_metrics.csv", rows)
    plot_heatmaps(rows, config, output)
    plot_method_comparison(rows, output)
    benefit = lambda c: (c["metrics"]["raw"]["absolute_error_hz"]
                         - c["metrics"]["full_wave_rectified"]["absolute_error_hz"])
    success = max(cases, key=benefit)
    failure_pool = [c for c in cases if not c["metrics"]["full_wave_rectified"]["recovered"]]
    failure = max(failure_pool, key=lambda c: c["metrics"]["full_wave_rectified"]["absolute_error_hz"])
    plot_am_case(success, output / "am_representative_success.png", "Rectification-helpful example")
    plot_am_case(failure, output / "am_representative_failure.png", "Rectification failure example")
    return rows, success, failure


def window_energy(frequencies, spectrum, center, half_width=0.15):
    return float(np.sum(spectrum[np.abs(frequencies-center) <= half_width]))


def entropy(spectrum):
    probability = spectrum / (np.sum(spectrum) + 1e-12)
    return float(-np.sum(probability * np.log(probability + 1e-12)) / np.log(len(probability)))


def analyze_real(trace_dir, output):
    per_band, per_case = [], []
    for metadata_path in sorted(trace_dir.glob("case_*_metadata.json")):
        case_id = metadata_path.name.replace("_metadata.json", "")
        metadata = json.loads(metadata_path.read_text())
        arrays = np.load(trace_dir / f"{case_id}_arrays.npz")
        f, spectra = arrays["frequencies_hz"], arrays["band_spectra"] ** 2
        cardiac = (f >= 0.5) & (f <= 3.0)
        fc = float(metadata["ecg_hr"]) / 60
        preferences = []
        for index, (band, spectrum) in enumerate(zip(arrays["bands_hz"], spectra)):
            local_f, local_s = f[cardiac], spectrum[cardiac]
            preferred = float(local_f[np.argmax(local_s)])
            preferences.append(preferred)
            target = window_energy(f, spectrum, fc)
            harmonics = sum(window_energy(f, spectrum, n*fc) for n in (2, 3) if n*fc <= 10)
            competitor_mask = cardiac & (np.abs(f-fc) > 0.3)
            competitor_f = f[competitor_mask]
            competitor_s = spectrum[competitor_mask]
            competitor = float(competitor_f[np.argmax(competitor_s)])
            competitor_energy = window_energy(f, spectrum, competitor)
            ratios = {"half": abs(preferred-0.5*fc), "fundamental": abs(preferred-fc),
                      "double": abs(preferred-2*fc), "triple": abs(preferred-3*fc)}
            family = min(ratios, key=ratios.get) if min(ratios.values()) <= 0.15 else "other"
            harmonic_score = (target + harmonics) / (1 + sum(n*fc <= 10 for n in (2, 3)))
            per_band.append({"case_id": case_id, "selection_class": metadata["selection_class"],
                             "ecg_hr": metadata["ecg_hr"], "heart_v6_hr": metadata["heart_v6_hr"],
                             "absolute_error_bpm": metadata["absolute_error_bpm"],
                             "band": f"{band[0]:g}-{band[1]:g}", "target_hz": fc,
                             "fundamental_energy": target, "harmonic_energy": harmonics,
                             "strongest_competing_hz": competitor,
                             "strongest_competing_energy": competitor_energy,
                             "target_to_distractor_ratio": target/(competitor_energy+1e-12),
                             "spectral_entropy": entropy(local_s),
                             "harmonic_score": harmonic_score,
                             "preferred_frequency_hz": preferred,
                             "preferred_harmonic_family": family})
        rounded = np.round(np.asarray(preferences) / 0.1) * 0.1
        modal = float(max(set(rounded), key=lambda x: (np.sum(rounded == x), -x)))
        cross_agreement = float(np.mean(np.abs(np.asarray(preferences)-modal) <= 0.15))
        target_agreement = float(np.mean(np.abs(np.asarray(preferences)-fc) <= 0.15))
        ratio = float(metadata["heart_v6_hr"])/float(metadata["ecg_hr"])
        subtype = "half_rate" if abs(ratio-.5) <= .15 else "double_rate" if abs(ratio-2) <= .2 else "other"
        per_case.append({"case_id": case_id, "selection_class": metadata["selection_class"],
                         "gross_subtype": subtype, "ecg_hr": metadata["ecg_hr"],
                         "heart_v6_hr": metadata["heart_v6_hr"],
                         "absolute_error_bpm": metadata["absolute_error_bpm"],
                         "modal_preferred_frequency_hz": modal,
                         "cross_band_agreement": cross_agreement,
                         "ecg_target_band_agreement": target_agreement,
                         "mean_target_to_distractor_ratio": np.mean([
                             float(r["target_to_distractor_ratio"]) for r in per_band if r["case_id"] == case_id]),
                         "mean_spectral_entropy": np.mean([
                             float(r["spectral_entropy"]) for r in per_band if r["case_id"] == case_id]),
                         "production_score_margin": metadata["heart_rate_result"]["diagnostics"]["score_margin"]})
    write_csv(output / "real_per_band_features.csv", per_band)
    write_csv(output / "real_case_features.csv", per_case)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [row["case_id"].replace("case_", "") for row in per_case]
    for axis, key, title in zip(axes,
            ("mean_target_to_distractor_ratio", "ecg_target_band_agreement", "mean_spectral_entropy"),
            ("Mean target/distractor", "Bands preferring ECG target", "Mean spectral entropy")):
        colors = ["#2ca02c" if r["selection_class"] == "low_error" else
                  "#ff7f0e" if r["selection_class"] == "moderate_error" else "#d62728" for r in per_case]
        axis.bar(labels, [float(r[key]) for r in per_case], color=colors)
        axis.set_title(title); axis.tick_params(axis="x", rotation=45); axis.grid(alpha=.2, axis="y")
    fig.tight_layout()
    fig.savefig(output / "real_success_failure_features.png", dpi=180)
    plt.close(fig)
    return per_band, per_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--am-config", type=Path, default=Path("configs/experiment1_am.json"))
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.am_config.read_text())
    am_rows, success, failure = analyze_am(config, args.output_dir)
    real_band, real_case = analyze_real(args.trace_dir, args.output_dir)
    summary = {
        "scope": "Experiment 1 representation analysis; not an ablation or biological AM claim",
        "am_rows": len(am_rows), "real_band_rows": len(real_band), "real_cases": len(real_case),
        "cross_band_agreement_definition": "fraction of six band preferred frequencies within 0.15 Hz of their 0.1-Hz-binned mode",
        "target_to_distractor_definition": "target-window spectral energy divided by strongest equal-width non-target window energy",
        "am_success_parameters": {k: success[k] for k in ("carrier_hz", "modulation_hz", "modulation_depth", "snr_db", "replicate")},
        "am_failure_parameters": {k: failure[k] for k in ("carrier_hz", "modulation_hz", "modulation_depth", "snr_db", "replicate")},
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
