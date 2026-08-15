"""Leakage-safe outer-LOSO evaluation and ablation for HeartV7 fusion."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_performance import metrics


ROOT = Path(__file__).resolve().parent
PARAMETER_GRID = [
    {
        "high_v6_conf_max": high_v6,
        "high_acf_conf_min": high_acf,
        "low_v6_conf_max": low_v6,
        "low_acf_conf_min": low_acf,
    }
    for high_v6, high_acf, low_v6, low_acf in itertools.product(
        (0.25, 0.30, 0.35), (0.60, 0.70, 0.80),
        (0.25, 0.30, 0.35), (0.20, 0.30, 0.40),
    )
]


def fusion_predictions(data, parameters, enable_high=True, enable_low=True):
    v6 = data.heart_v6_hr.to_numpy(dtype=float)
    acf = data.autocorrelation_hr.to_numpy(dtype=float)
    ratio = v6 / acf
    high = (
        enable_high & (ratio >= 1.9) & (ratio <= 2.1)
        & (data.heart_v6_confidence.to_numpy() < parameters["high_v6_conf_max"])
        & (data.autocorrelation_confidence.to_numpy() > parameters["high_acf_conf_min"])
    )
    low = (
        enable_low & (ratio >= 0.6) & (ratio <= 0.8)
        & (data.heart_v6_confidence.to_numpy() < parameters["low_v6_conf_max"])
        & (data.autocorrelation_confidence.to_numpy() > parameters["low_acf_conf_min"])
    )
    switched = high | low
    return np.where(switched, acf, v6), high, low


def subject_macro_mae(data, predictions):
    errors = np.abs(predictions - data.ecg_hr.to_numpy(dtype=float))
    subjects = data.subject_id.to_numpy()
    return float(np.mean([np.mean(errors[subjects == subject]) for subject in np.unique(subjects)]))


def select_parameters(development):
    candidates = []
    for parameters in PARAMETER_GRID:
        predictions, high, low = fusion_predictions(development, parameters)
        candidates.append((subject_macro_mae(development, predictions), int(np.sum(high | low)), parameters))
    return min(candidates, key=lambda item: (item[0], item[1]))


def cluster_bootstrap_delta(data, method_a, method_b, iterations=10000, seed=42):
    """Bootstrap subjects, retaining all samples within each resampled cluster."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(data.subject_id)
    per_subject = {}
    for subject in subjects:
        rows = data[data.subject_id == subject]
        per_subject[subject] = (
            np.mean(np.abs(rows[method_a] - rows.ecg_hr)),
            np.mean(np.abs(rows[method_b] - rows.ecg_hr)),
        )
    deltas = np.empty(iterations)
    for index in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        deltas[index] = np.mean([per_subject[subject][1] - per_subject[subject][0] for subject in sampled])
    return {
        "mean_macro_mae_delta_b_minus_a": float(np.mean(deltas)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
        "probability_b_better": float(np.mean(deltas < 0)),
        "bootstrap_iterations": iterations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "baselines" / "per_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "heart_v7_protocol")
    args = parser.parse_args(argv)
    data = pd.read_csv(args.input)
    subjects = sorted(data.subject_id.unique())
    predictions = pd.Series(index=data.index, dtype=float)
    fold_records = []

    for fold, test_subject in enumerate(subjects):
        development = data[data.subject_id != test_subject]
        test = data[data.subject_id == test_subject]
        development_score, development_switches, parameters = select_parameters(development)
        test_predictions, high, low = fusion_predictions(test, parameters)
        predictions.loc[test.index] = test_predictions
        fold_records.append({
            "fold": fold,
            "test_subject": int(test_subject),
            "test_samples": int(len(test)),
            "development_macro_mae": development_score,
            "development_switches": development_switches,
            "test_switches_high": int(np.sum(high)),
            "test_switches_low": int(np.sum(low)),
            "test_v6_mae": float(np.mean(np.abs(test.heart_v6_hr - test.ecg_hr))),
            "test_v7_oof_mae": float(np.mean(np.abs(test_predictions - test.ecg_hr))),
            **parameters,
        })

    data["heart_v7_oof_hr"] = predictions
    fixed = {
        "high_v6_conf_max": 0.35, "high_acf_conf_min": 0.80,
        "low_v6_conf_max": 0.35, "low_acf_conf_min": 0.20,
    }
    high_only, _, _ = fusion_predictions(data, fixed, enable_high=True, enable_low=False)
    low_only, _, _ = fusion_predictions(data, fixed, enable_high=False, enable_low=True)
    full_fixed, _, _ = fusion_predictions(data, fixed)
    data["heart_v7_high_only_hr"] = high_only
    data["heart_v7_low_only_hr"] = low_only
    data["heart_v7_fixed_hr"] = full_fixed

    methods = ["heart_v6_hr", "heart_v7_high_only_hr", "heart_v7_low_only_hr",
               "heart_v7_fixed_hr", "heart_v7_oof_hr"]
    report = {
        "protocol": "outer leave-one-subject-out; parameters minimize subject-macro MAE on the other nine subjects",
        "parameter_candidates": len(PARAMETER_GRID),
        "overall": {method: metrics(data.ecg_hr, data[method]) for method in methods},
        "subject_macro_mae": {
            method: subject_macro_mae(data, data[method].to_numpy()) for method in methods
        },
        "folds": fold_records,
        "cluster_bootstrap_v6_vs_v7_oof": cluster_bootstrap_delta(
            data, "heart_v6_hr", "heart_v7_oof_hr"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    with (args.output_dir / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_records[0].keys())
        writer.writeheader(); writer.writerows(fold_records)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
