"""Audit HeartV6 errors, reference disagreement, and confidence/coverage trade-offs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def classify_error(reference, estimate):
    error = estimate - reference
    ratio = estimate / reference
    if abs(error) <= 5:
        return "within_5_bpm"
    if 1.8 <= ratio <= 2.2:
        return "double_rate"
    if 0.42 <= ratio <= 0.58:
        return "half_rate"
    if abs(error) > 10:
        return "gross_other"
    return "moderate_other"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "reports" / "baselines" / "per_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "failure_analysis")
    args = parser.parse_args(argv)
    data = pd.read_csv(args.input)
    data["heart_v6_error"] = data["heart_v6_hr"] - data["ecg_hr"]
    data["heart_v6_abs_error"] = data["heart_v6_error"].abs()
    data["heart_v6_ratio"] = data["heart_v6_hr"] / data["ecg_hr"]
    data["error_class"] = [classify_error(ref, est) for ref, est in zip(data.ecg_hr, data.heart_v6_hr)]
    data["ecg_legacy_disagreement"] = (data["ecg_hr_legacy"] - data["ecg_hr"]).abs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worst = data.nlargest(200, "heart_v6_abs_error")
    worst.to_csv(args.output_dir / "worst_200.csv", index=False)
    reference_audit = data.nlargest(200, "ecg_legacy_disagreement")
    reference_audit.to_csv(args.output_dir / "reference_disagreement_200.csv", index=False)

    coverage_rows = []
    for threshold in np.linspace(0, 0.5, 51):
        accepted = data[data.heart_v6_confidence >= threshold]
        if not len(accepted):
            continue
        coverage_rows.append({
            "confidence_threshold": float(threshold),
            "accepted": int(len(accepted)),
            "coverage_percent": float(100 * len(accepted) / len(data)),
            "mae_bpm": float(accepted.heart_v6_abs_error.mean()),
            "within_3_bpm_percent": float(100 * (accepted.heart_v6_abs_error <= 3).mean()),
            "within_5_bpm_percent": float(100 * (accepted.heart_v6_abs_error <= 5).mean()),
        })
    with (args.output_dir / "coverage_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_rows[0].keys())
        writer.writeheader()
        writer.writerows(coverage_rows)

    class_counts = data.error_class.value_counts().to_dict()
    subject_summary = {}
    for subject, rows in data.groupby("subject_id"):
        subject_summary[str(subject)] = {
            "n": int(len(rows)),
            "mae_bpm": float(rows.heart_v6_abs_error.mean()),
            "within_5_bpm_percent": float(100 * (rows.heart_v6_abs_error <= 5).mean()),
            "mean_confidence": float(rows.heart_v6_confidence.mean()),
            "error_classes": rows.error_class.value_counts().to_dict(),
        }
    report = {
        "n": int(len(data)),
        "error_classes": class_counts,
        "legacy_reference_disagreement_mae_bpm": float(data.ecg_legacy_disagreement.mean()),
        "legacy_reference_disagreement_over_5_bpm_percent": float(100 * (data.ecg_legacy_disagreement > 5).mean()),
        "confidence_error_spearman": float(data[["heart_v6_confidence", "heart_v6_abs_error"]].corr(method="spearman").iloc[0, 1]),
        "by_subject": subject_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "by_subject"}, indent=2))


if __name__ == "__main__":
    main()
