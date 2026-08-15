"""Evaluate classical BCG heart-rate baselines by subject and sensor distance."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.ECG_heartrate_alg import DetectionECGPeaks_TenSond
from Algorithm.Filters import BandPassFilter
from Algorithm.heart_rate import autocorrelation_heart_rate, fft_peak_heart_rate, heart_v6, heart_v7
from Algorithm.ecg_reference import detect_ecg_r_peaks
from Dataset.metadata import parse_real_recording_name
from validate_performance import DEFAULT_DATA, metrics, safe_heart_rate


PROJECT_ROOT = Path(__file__).resolve().parent
ESTIMATORS = {
    "fft_peak": fft_peak_heart_rate,
    "autocorrelation": autocorrelation_heart_rate,
    "heart_v6": heart_v6,
    "heart_v7": heart_v7,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "baselines")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def result_metrics(rows, method):
    return metrics([row["ecg_hr"] for row in rows], [row[f"{method}_hr"] for row in rows])


def grouped_metrics(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        group: {method: result_metrics(group_rows, method) for method in ESTIMATORS}
        for group, group_rows in sorted(groups.items(), key=lambda item: float(item[0]))
    }


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.data_dir.glob("*.npy"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")
    rows = []
    malformed = []
    for path in tqdm(files, desc="Evaluating baselines"):
        try:
            metadata = parse_real_recording_name(path)
            pair = np.load(path)
            if pair.shape != (2, 1000) or not np.isfinite(pair).all():
                raise ValueError(f"invalid array shape or values: {pair.shape}")
            bcg = zscore_normalize(BandPassFilter(pair[0], 0.5, 25, 4, 100, padlen=500))
            ecg = zscore_normalize(BandPassFilter(pair[1], 0.5, 25, 4, 100, padlen=500))
            ecg_reference = detect_ecg_r_peaks(ecg, fs=100)
            row = {
                **metadata.to_dict(),
                "ecg_hr": ecg_reference.bpm if ecg_reference.valid else math.nan,
                "ecg_confidence": ecg_reference.confidence,
                "ecg_interval_cv": ecg_reference.interval_cv,
                "ecg_peak_count": len(ecg_reference.peaks),
                "ecg_hr_legacy": safe_heart_rate(DetectionECGPeaks_TenSond, ecg, k=1.5, p=500, sfreq=100),
            }
            for name, estimator in ESTIMATORS.items():
                try:
                    result = estimator(bcg)
                    row[f"{name}_hr"] = result.bpm if result.valid else math.nan
                    row[f"{name}_confidence"] = result.confidence
                except (TypeError, ValueError, ZeroDivisionError, FloatingPointError, IndexError) as error:
                    row[f"{name}_hr"] = math.nan
                    row[f"{name}_confidence"] = 0.0
                    row[f"{name}_error"] = str(error)
            rows.append(row)
        except (OSError, ValueError) as error:
            malformed.append({"filename": path.name, "error": str(error)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "files_seen": len(files),
        "files_evaluated": len(rows),
        "malformed": malformed,
        "overall": {method: result_metrics(rows, method) for method in ESTIMATORS},
        "by_subject": grouped_metrics(rows, "subject_id"),
        "by_distance_cm": grouped_metrics(rows, "distance_cm"),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_rows = []
    for subject, methods in report["by_subject"].items():
        for method, values in methods.items():
            summary_rows.append({"subject_id": subject, "method": method, **values})
    with (args.output_dir / "by_subject.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"files_evaluated": len(rows), "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
