"""Evaluate raw and model-filtered BCG heart rate against reference ECG."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from Algorithm.BCG_heartrate_alg import HeartV6_TenSecond
from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.ECG_heartrate_alg import DetectionECGPeaks_TenSond
from Algorithm.Filters import BandPassFilter
from Model.LSTM import LSTM_BCGFilter_Pre


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT.parent / "Dataset" / "BCG" / "DeepLearningData" / "BCG_ECG_10sec"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weight" / "BCG_HeartFilter" / "260109" / "best_Test_model.pth"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "real_world")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def valid_hr(value):
    return np.isfinite(value) and 30 <= value <= 220


def metrics(reference, estimate):
    ref, est = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(est)
    attempted = int(len(ref))
    ref, est = ref[mask], est[mask]
    if not len(ref):
        return {"n": 0, "attempted": attempted, "failure_rate_percent": 100.0}
    error = est - ref
    result = {
        "n": int(len(ref)), "attempted": attempted,
        "failure_rate_percent": float(100 * (1 - len(ref) / attempted)) if attempted else 0.0,
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error ** 2))), "bias_bpm": float(np.mean(error)),
        "within_3_bpm_percent": float(100 * np.mean(np.abs(error) <= 3)),
        "within_5_bpm_percent": float(100 * np.mean(np.abs(error) <= 5)),
    }
    can_correlate = len(ref) > 1 and np.std(ref) > 0 and np.std(est) > 0
    result["correlation"] = float(np.corrcoef(ref, est)[0, 1]) if can_correlate else None
    sd = float(np.std(error, ddof=1)) if len(ref) > 1 else 0.0
    result.update({"sd_bpm": sd, "loa_lower_bpm": result["bias_bpm"] - 1.96 * sd,
                   "loa_upper_bpm": result["bias_bpm"] + 1.96 * sd})
    return result


def bland_altman_plot(reference, estimate, title, path):
    ref, est = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[mask], est[mask]
    stats = metrics(ref, est)
    diff = est - ref
    plt.figure(figsize=(7, 5))
    plt.scatter((ref + est) / 2, diff, s=12, alpha=0.6)
    for value, style in ((stats["bias_bpm"], "-"), (stats["loa_upper_bpm"], "--"),
                         (stats["loa_lower_bpm"], "--")):
        plt.axhline(value, linestyle=style)
    plt.title(title)
    plt.xlabel("Mean HR (BPM)")
    plt.ylabel("BCG − ECG (BPM)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def safe_heart_rate(function, signal, **kwargs):
    try:
        value = float(function(signal, **kwargs))
        return value if valid_hr(value) else math.nan
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError, IndexError):
        return math.nan


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.data_dir.glob("*.npy"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = LSTM_BCGFilter_Pre(1000, 1, 128, 1, 0.2, 6, True, 1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    rows = []

    for path in tqdm(files, desc="Evaluating"):
        pair = np.load(path)
        if pair.shape != (2, 1000) or not np.isfinite(pair).all():
            continue
        bcg = zscore_normalize(BandPassFilter(pair[0], 0.5, 25, 4, 100, padlen=500))
        ecg = zscore_normalize(BandPassFilter(pair[1], 0.5, 25, 4, 100, padlen=500))
        tensor = torch.as_tensor(bcg, dtype=torch.float32, device=device)[None, None]
        with torch.inference_mode():
            filtered = model(tensor).squeeze().cpu().numpy()
        rows.append({
            "file": path.name,
            "ecg_hr": safe_heart_rate(DetectionECGPeaks_TenSond, ecg, k=1.5, p=500, sfreq=100),
            "raw_bcg_hr": safe_heart_rate(HeartV6_TenSecond, bcg),
            "filtered_bcg_hr": safe_heart_rate(HeartV6_TenSecond, filtered),
        })

    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reference = [row["ecg_hr"] for row in rows]
    raw = [row["raw_bcg_hr"] for row in rows]
    filtered = [row["filtered_bcg_hr"] for row in rows]
    report = {
        "files_seen": len(files), "files_evaluated": len(rows),
        "raw_bcg": metrics(reference, raw), "filtered_bcg": metrics(reference, filtered),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bland_altman_plot(reference, raw, "ECG vs Raw BCG", args.output_dir / "raw_bcg_bland_altman.png")
    bland_altman_plot(reference, filtered, "ECG vs Filtered BCG", args.output_dir / "filtered_bcg_bland_altman.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
