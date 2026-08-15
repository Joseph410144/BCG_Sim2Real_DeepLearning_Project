"""Outer-LOSO calibration of HeartV6 confidence as P(error <= 5 BPM)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from Algorithm.confidence import BinnedIsotonicCalibrator


ROOT = Path(__file__).resolve().parent


def expected_calibration_error(probabilities, labels, bins=10):
    boundaries = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    rows = []
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        count = int(np.sum(mask))
        value += count / total * abs(confidence - accuracy)
        rows.append({"low": float(low), "high": float(high), "n": count,
                     "confidence": confidence, "accuracy": accuracy})
    return float(value), rows


def binary_auc(probabilities, labels):
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = np.sum(labels), np.sum(~labels)
    if not positives or not negatives:
        return None
    ranks = rankdata(probabilities)
    return float((np.sum(ranks[labels]) - positives * (positives + 1) / 2) / (positives * negatives))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "baselines" / "per_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "confidence_calibration")
    args = parser.parse_args(argv)
    data = pd.read_csv(args.input)
    data["within_5"] = (data.heart_v6_hr - data.ecg_hr).abs() <= 5
    data["calibrated_probability"] = np.nan
    fold_models = []
    for test_subject in sorted(data.subject_id.unique()):
        development = data[data.subject_id != test_subject]
        test = data[data.subject_id == test_subject]
        calibrator = BinnedIsotonicCalibrator(bins=20).fit(
            development.heart_v6_confidence, development.within_5)
        data.loc[test.index, "calibrated_probability"] = calibrator.predict(test.heart_v6_confidence)
        fold_models.append({"test_subject": int(test_subject),
                            "score_knots": calibrator.x_.tolist(),
                            "probability_knots": calibrator.y_.tolist()})
    probabilities = data.calibrated_probability.to_numpy()
    labels = data.within_5.to_numpy(dtype=float)
    ece, reliability = expected_calibration_error(probabilities, labels)
    report = {
        "protocol": "outer leave-one-subject-out calibration",
        "target": "absolute HR error <= 5 BPM",
        "positive_rate": float(np.mean(labels)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error": ece,
        "auc": binary_auc(probabilities, labels),
        "reliability_bins": reliability,
        "fold_models": fold_models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "out_of_fold_calibration.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "fold_models"}, indent=2))


if __name__ == "__main__":
    main()
