"""Create a reproducible nested leave-one-subject-out split manifest."""

import argparse
import json
from pathlib import Path

from Dataset.metadata import parse_real_recording_name
from Dataset.splits import leave_one_subject_out_folds
from validate_performance import DEFAULT_DATA


PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "subject_folds.json")
    args = parser.parse_args(argv)
    metadata = [parse_real_recording_name(path) for path in sorted(args.data_dir.glob("*.npy"))]
    if not metadata:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")
    folds = leave_one_subject_out_folds(item.subject_id for item in metadata)
    counts = {}
    for item in metadata:
        counts[str(item.subject_id)] = counts.get(str(item.subject_id), 0) + 1
    manifest = {
        "protocol": "nested_leave_one_subject_out",
        "selection_rule": "Validation subject is the next subject ID cyclically; test subject is never used for selection.",
        "subject_sample_counts": counts,
        "folds": [fold.to_dict() for fold in folds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
