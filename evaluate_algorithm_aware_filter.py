"""Evaluate raw versus residual-filtered HR on a fold's held-out test subject."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Algorithm.heart_rate import heart_v6, heart_v7
from Dataset.BCG_Dataset import RealBCGHeartRateDataset
from Model.AlgorithmAwareFilter import AlgorithmAwareResidualFilter
from train_BCG_HeartFilter import resolve_device
from train_algorithm_aware_filter import DEFAULT_REAL_DATA
from validate_performance import metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--real-data-dir", type=Path, default=DEFAULT_REAL_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = args.run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    config = json.loads(config_path.read_text())
    test_subjects = config["subjects"]["test_subjects"]
    if set(test_subjects) & (set(config["subjects"]["train_subjects"]) | set(config["subjects"]["val_subjects"])):
        raise ValueError("Subject leakage detected in run config")
    weights = args.weights or args.run_dir / "best_real.pth"
    if not weights.is_file():
        weights = args.run_dir / "last_model.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"No model weights found in {args.run_dir}")
    output_dir = args.output_dir or args.run_dir / "test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = AlgorithmAwareResidualFilter(
        config["channels"], config["blocks"],
        initial_gate=config.get("initial_gate", -4.0),
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    dataset = RealBCGHeartRateDataset(str(args.real_data_dir), test_subjects)
    if args.limit is not None and args.limit < len(dataset): dataset = Subset(dataset, range(args.limit))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.inference_mode():
        for inputs, target_bpm, subjects, filenames in tqdm(data_loader, desc="Held-out evaluation"):
            filtered = model(inputs.to(device, dtype=torch.float32)).cpu().numpy()
            raw = inputs.numpy()
            for index, filename in enumerate(filenames):
                raw_v6, raw_v7 = heart_v6(raw[index, 0]), heart_v7(raw[index, 0])
                filtered_v6, filtered_v7 = heart_v6(filtered[index, 0]), heart_v7(filtered[index, 0])
                rows.append({
                    "filename": filename, "subject_id": int(subjects[index]),
                    "ecg_hr": float(target_bpm[index]),
                    "raw_v6_hr": raw_v6.bpm, "raw_v7_hr": raw_v7.bpm,
                    "filtered_v6_hr": filtered_v6.bpm, "filtered_v7_hr": filtered_v7.bpm,
                    "raw_v6_confidence": raw_v6.confidence, "filtered_v6_confidence": filtered_v6.confidence,
                    "mean_absolute_residual": float(np.mean(np.abs(filtered[index, 0] - raw[index, 0]))),
                })
    with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    reference = [row["ecg_hr"] for row in rows]
    report = {"weights": str(weights.resolve()), "test_subjects": test_subjects, "n": len(rows)}
    for method in ("raw_v6_hr", "raw_v7_hr", "filtered_v6_hr", "filtered_v7_hr"):
        report[method] = metrics(reference, [row[method] for row in rows])
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
