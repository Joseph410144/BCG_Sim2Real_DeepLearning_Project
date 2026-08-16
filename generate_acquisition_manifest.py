"""Generate and validate a private manifest for repeated personal BCG recordings."""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from Dataset.provenance import build_source_index, manifest_entry, validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WINDOWS = PROJECT_ROOT.parent / "Dataset" / "BCG" / "DeepLearningData" / "BCG_ECG_10sec"
DEFAULT_SOURCES = PROJECT_ROOT.parent / "Dataset" / "BCG" / "DeepLearningData" / "BCG_ECG_database"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-dir", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "acquisition_manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-content-verification", action="store_true")
    return parser.parse_args(argv)


def git_version():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN", True


def main(argv=None):
    args = parse_args(argv)
    windows = sorted(args.window_dir.glob("*.npy"))
    if args.limit is not None:
        windows = windows[:args.limit]
    if not windows:
        raise FileNotFoundError(f"No window files found: {args.window_dir}")
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_index = build_source_index(args.source_root)
    rows = [manifest_entry(path, args.source_root, source_index,
                           verify_content=not args.skip_content_verification)
            for path in windows]
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    validation = validate_manifest(rows)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    revision, dirty = git_version()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision, "code_dirty": dirty,
        "dataset_interpretation": "single_person_repeated_recordings",
        "person_count": 1,
        "window_dir": str(args.window_dir.resolve()),
        "source_root": str(args.source_root.resolve()),
        "content_verification": not args.skip_content_verification,
        "limit": args.limit,
        "window_count": len(rows),
        "validation_summary": {key: value for key, value in validation.items() if key != "issues"},
        "privacy": "Contains file-level metadata and checksums; keep generated manifest private.",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if validation["errors"]:
        raise SystemExit(f"Manifest validation found {validation['errors']} error(s)")


if __name__ == "__main__":
    main()
