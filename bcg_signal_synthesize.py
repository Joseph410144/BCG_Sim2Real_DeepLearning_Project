"""Generate deterministic train/validation/test synthetic BCG datasets."""

import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Algorithm.Bcg_signal_synthesis_function import BedBCGGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "Dataset" / "BCG" / "Synthesis"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100_000, help="Total number across all splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sampling-rate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def split_counts(total, ratios):
    if total < 1 or any(r < 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("count must be positive and split ratios must be non-negative and sum to 1")
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    return train, val, total - train - val


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    counts = split_counts(args.count, (args.train_ratio, args.val_ratio, args.test_ratio))
    generator = BedBCGGenerator(fs=args.sampling_rate)

    for split, count in zip(("training", "validation", "test"), counts):
        output_dir = args.output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in tqdm(range(count), desc=f"Generating {split}"):
            path = output_dir / f"{split}_data_{index}.npy"
            if path.exists() and not args.overwrite:
                continue
            noisy_bcg, clean_bcg = generator.generate(duration=args.duration)
            np.save(path, np.asarray([noisy_bcg, clean_bcg], dtype=np.float32))


if __name__ == "__main__":
    main()
