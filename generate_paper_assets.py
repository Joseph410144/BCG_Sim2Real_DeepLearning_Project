"""Generate reproducible figures and tables used by the LaTeX progress paper."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "paper" / "figures"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((REPORTS / "baselines" / "metrics.json").read_text())
    protocol = json.loads((REPORTS / "heart_v7_protocol" / "metrics.json").read_text())
    methods = ["fft_peak", "autocorrelation", "heart_v6"]
    labels = ["FFT peak", "Autocorrelation", "HeartV6", "HeartV7 OOF"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    values = {
        "MAE (BPM)": [report["overall"][method]["mae_bpm"] for method in methods] + [protocol["overall"]["heart_v7_oof_hr"]["mae_bpm"]],
        "RMSE (BPM)": [report["overall"][method]["rmse_bpm"] for method in methods] + [protocol["overall"]["heart_v7_oof_hr"]["rmse_bpm"]],
        "Within $\\pm$5 BPM (%)": [report["overall"][method]["within_5_bpm_percent"] for method in methods] + [protocol["overall"]["heart_v7_oof_hr"]["within_5_bpm_percent"]],
    }
    colors = ["#9aa0a6", "#5f9ea0", "#2f6f9f", "#d95f02"]
    for axis, (title, data) in zip(axes, values.items()):
        bars = axis.bar(labels, data, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, data):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}",
                      ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "overall_performance.pdf", bbox_inches="tight")
    plt.close(fig)

    oof = pd.read_csv(REPORTS / "heart_v7_protocol" / "out_of_fold_predictions.csv")
    subjects = sorted(report["by_subject"], key=int)
    x = np.arange(len(subjects))
    v6 = [report["by_subject"][subject]["heart_v6"]["mae_bpm"] for subject in subjects]
    v7 = [float(np.mean(np.abs(oof[oof.subject_id == int(subject)].heart_v7_oof_hr -
                                    oof[oof.subject_id == int(subject)].ecg_hr))) for subject in subjects]
    fig, axis = plt.subplots(figsize=(8, 3.6))
    width = 0.38
    axis.bar(x - width / 2, v6, width, label="HeartV6", color="#2f6f9f")
    axis.bar(x + width / 2, v7, width, label="HeartV7", color="#d95f02")
    axis.set_xticks(x, subjects)
    axis.set_xlabel("Subject ID")
    axis.set_ylabel("MAE (BPM)")
    axis.set_title("Subject-wise performance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "subject_mae.pdf", bbox_inches="tight")
    plt.close(fig)

    coverage = pd.read_csv(REPORTS / "failure_analysis" / "coverage_accuracy.csv")
    fig, axis = plt.subplots(figsize=(5.5, 3.6))
    axis.plot(coverage.coverage_percent, coverage.mae_bpm, marker="o", markersize=2.5,
              color="#2f6f9f")
    axis.set_xlabel("Coverage (%)")
    axis.set_ylabel("MAE (BPM)")
    axis.set_title("HeartV6 confidence--coverage trade-off")
    axis.invert_xaxis()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "coverage_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    data = pd.read_csv(REPORTS / "baselines" / "per_sample.csv")
    fig, axis = plt.subplots(figsize=(5.5, 3.6))
    axis.scatter(data.ecg_hr, data.ecg_hr_legacy, s=4, alpha=0.18, color="#5f9ea0")
    limits = [min(data.ecg_hr.min(), data.ecg_hr_legacy.min()),
              max(data.ecg_hr.max(), data.ecg_hr_legacy.max())]
    axis.plot(limits, limits, "--", color="black", linewidth=1)
    axis.set_xlim(limits); axis.set_ylim(limits)
    axis.set_xlabel("Audited ECG reference (BPM)")
    axis.set_ylabel("Legacy ECG detector (BPM)")
    axis.set_title("Reference-label audit")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "reference_audit.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
