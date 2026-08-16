"""Generate aggregate, publication-safe figures for the manuscript."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

COLORS = ("#2F5D8A", "#B64E5A", "#3B7D5A", "#777777")
METHOD_LABELS = {
    "raw": "Raw spectrum",
    "full_wave_rectified": "Full-wave rectified",
    "hilbert_envelope": "Hilbert envelope",
    "squared_energy": "Squared energy",
}
FAMILY_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "clipping": "Clipping",
    "contiguous_sample_loss": "Contiguous loss",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def polish_axis(axis, grid_axis="both"):
    axis.grid(alpha=0.18, linewidth=0.55, axis=grid_axis)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        OUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def am_figure():
    rows = read_csv(
        ROOT
        / "reports/track_a/experiment1_analysis_20260816_v2/am_quantitative_metrics.csv"
    )
    methods = ("raw", "full_wave_rectified", "hilbert_envelope", "squared_energy")
    snrs = ("inf", "20", "10", "0")
    depths = ("0.1", "0.3", "0.6")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    for method, color in zip(methods, COLORS):
        values = [
            100
            * np.mean(
                [
                    float(row["recovered"])
                    for row in rows
                    if row["method"] == method and row["snr_db"] == snr
                ]
            )
            for snr in snrs
        ]
        axes[0].plot(
            range(len(snrs)),
            values,
            marker="o",
            color=color,
            label=METHOD_LABELS[method],
        )
    axes[0].set(
        xlabel="SNR (dB; NF = noise-free)",
        ylabel="Recovery rate (%)",
        xticks=range(len(snrs)),
        xticklabels=("NF", "20", "10", "0"),
        ylim=(-3, 103),
    )
    axes[0].legend(frameon=False, ncol=2, handlelength=1.6)
    polish_axis(axes[0])

    rectified = [row for row in rows if row["method"] == "full_wave_rectified"]
    grid = np.array(
        [
            [
                100
                * np.mean(
                    [
                        float(row["recovered"])
                        for row in rectified
                        if row["modulation_depth"] == depth
                        and row["snr_db"] == snr
                    ]
                )
                for snr in snrs
            ]
            for depth in depths
        ]
    )
    image = axes[1].imshow(
        grid, aspect="auto", vmin=0, vmax=100, cmap="viridis", interpolation="none"
    )
    axes[1].set(
        xticks=range(4),
        xticklabels=("NF", "20", "10", "0"),
        yticks=range(3),
        yticklabels=depths,
        xlabel="SNR (dB)",
        ylabel="Modulation depth",
    )
    for row_index in range(3):
        for column_index in range(4):
            value = grid[row_index, column_index]
            color = "white" if value < 45 or value > 75 else "black"
            axes[1].text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=axes[1], label="Recovery rate (%)", shrink=0.82)
    colorbar.ax.tick_params(labelsize=7)
    save(fig, "am_mechanism")


def failure_figure():
    base = ROOT / "reports/track_a/experiment2_failure_stages_20260816_v3"
    summary = json.load((base / "summary.json").open(encoding="utf-8"))
    labels = ("Representation", "Fusion/scoring", "Refinement", "Mixed/unresolved")
    keys = (
        "representation_failure",
        "fusion_scoring_failure",
        "refinement_failure",
        "unresolved_mixed",
    )
    counts = np.array([summary["failure_stages"][key]["n"] for key in keys])
    percentages = 100 * counts / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    x_positions = np.arange(len(labels))
    axes[0].bar(x_positions, percentages, color=COLORS, width=0.68)
    axes[0].set(
        ylabel="Material failures (%)",
        ylim=(0, 42),
        xticks=x_positions,
        xticklabels=("Representation", "Fusion /\nscoring", "Refinement", "Mixed /\nunresolved"),
    )
    for index, value in enumerate(percentages):
        axes[0].text(index, value + 0.7, f"{value:.1f}%", ha="center", fontsize=7)
    polish_axis(axes[0], "y")

    rows = read_csv(base / "by_spectral_error_severity.csv")
    x_positions = np.arange(len(rows))
    axes[1].bar(
        x_positions - 0.18,
        [float(row["mae"]) for row in rows],
        0.36,
        color=COLORS[0],
        label="Spectral estimate",
    )
    axes[1].bar(
        x_positions + 0.18,
        [float(row["final_mae"]) for row in rows],
        0.36,
        color=COLORS[1],
        label="After interval refinement",
    )
    axes[1].set(
        xticks=x_positions,
        xticklabels=[row["spectral_error_stratum"] for row in rows],
        xlabel="Spectral-error stratum (BPM)",
        ylabel="MAE (BPM)",
    )
    axes[1].legend(frameon=False)
    polish_axis(axes[1], "y")
    save(fig, "failure_mechanisms")


def degradation_figure():
    rows = read_csv(
        ROOT / "reports/track_a/experiment3_robustness_20260816_v2/paired_summary.csv"
    )
    families = ("gaussian_noise", "clipping", "contiguous_sample_loss")
    severities = ("mild", "moderate", "severe")
    specifications = (
        ("mean_delta_mean_target_to_distractor_ratio", r"$\Delta$ target/distractor ratio"),
        ("mean_delta_mean_spectral_entropy", r"$\Delta$ spectral entropy"),
        ("mean_delta_cross_band_agreement", r"$\Delta$ cross-band agreement"),
        ("mean_delta_harmonic_family_stability", r"$\Delta$ family stability"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55), constrained_layout=True)
    for axis, (key, label) in zip(axes.ravel(), specifications):
        for family, color in zip(families, COLORS):
            values = [
                float(
                    next(
                        row
                        for row in rows
                        if row["perturbation"] == family
                        and row["severity"] == severity
                    )[key]
                )
                for severity in severities
            ]
            axis.plot(
                range(3),
                values,
                marker="o",
                color=color,
                label=FAMILY_LABELS[family],
            )
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set(
            xticks=range(3),
            xticklabels=("Mild", "Moderate", "Severe"),
            ylabel=label,
        )
        axis.margins(x=0.08)
        polish_axis(axis)
    axes[0, 0].legend(frameon=False, ncol=1, loc="best")
    save(fig, "representation_degradation")


def reconstruction_figure():
    main_triplets = read_csv(
        ROOT / "reports/track_a/experiment4_nonlearned_20260816_v1/triplets.csv"
    )
    corrected_triplets = read_csv(
        ROOT / "reports/track_a/experiment4_pchip_correction_20260816_v1/triplets.csv"
    )
    triplets = [
        row for row in main_triplets if row["method"] != "pchip_interpolation"
    ] + corrected_triplets
    conditions = (
        ("gaussian_noise", "bandlimit_0p5_12hz", "Noise\nband-limit"),
        ("gaussian_noise", "wiener_5", "Noise\nWiener"),
        ("clipping", "saturation_linear_interpolation", "Clipping\nlinear"),
        ("contiguous_sample_loss", "linear_interpolation", "Loss\nlinear"),
        ("contiguous_sample_loss", "pchip_interpolation", "Loss\nPCHIP"),
    )

    degraded = []
    recovered = []
    for family, method, _ in conditions:
        selected = [
            row
            for row in triplets
            if row["perturbation"] == family
            and row["severity"] == "severe"
            and row["method"] == method
        ]
        degraded.append(
            np.mean([float(row["degraded_final_error_bpm"]) for row in selected])
        )
        recovered.append(np.mean([float(row["final_error_bpm"]) for row in selected]))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    x_positions = np.arange(len(conditions))
    axes[0].bar(
        x_positions - 0.18,
        degraded,
        0.36,
        color=COLORS[0],
        label="Degraded",
    )
    axes[0].bar(
        x_positions + 0.18,
        recovered,
        0.36,
        color=COLORS[1],
        label="Recovered",
    )
    axes[0].set(
        xticks=x_positions,
        xticklabels=[condition[2] for condition in conditions],
        ylabel="Downstream MAE (BPM)",
    )
    axes[0].legend(frameon=False)
    polish_axis(axes[0], "y")

    sample = triplets[:: max(1, len(triplets) // 15000)]
    axes[1].scatter(
        [float(row["waveform_recovery_fraction"]) for row in sample],
        [
            float(row["degraded_final_error_bpm"]) - float(row["final_error_bpm"])
            for row in sample
        ],
        s=5,
        color=COLORS[0],
        alpha=0.10,
        linewidths=0,
        rasterized=True,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.7)
    axes[1].set(
        xlabel="Waveform RMSE recovery fraction",
        ylabel="HR-error improvement (BPM)",
        xlim=(-3, 1),
    )
    polish_axis(axes[1])
    save(fig, "reconstruction_task_mismatch")


if __name__ == "__main__":
    am_figure()
    failure_figure()
    degradation_figure()
    reconstruction_figure()
