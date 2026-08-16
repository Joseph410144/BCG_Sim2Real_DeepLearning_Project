"""Provenance helpers for the single-person repeated-recording BCG dataset."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from Dataset.metadata import parse_real_recording_name


UNKNOWN = "UNKNOWN"
WINDOW_SAMPLES = 1000
PERSON_ID = "PERSON_001"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_basename(window_filename):
    """Undo the preprocessing script's final ``_<window>.npy`` suffix."""
    path = Path(window_filename)
    metadata = parse_real_recording_name(path.name)
    suffix = f"_{metadata.segment_id}.npy"
    if not path.name.endswith(suffix):
        raise ValueError(f"Cannot derive source basename: {path.name}")
    return path.name[:-len(suffix)] + ".npy"


def build_source_index(source_root):
    index = defaultdict(list)
    for path in sorted(Path(source_root).rglob("*_heart.npy")):
        index[path.name].append(path)
    return index


def stable_source_id(source_sha256):
    """Use content identity so copied source files remain one dependency group."""
    return f"source_{source_sha256[:16]}"


def manifest_entry(window_path, source_root, source_index=None, verify_content=True):
    window_path = Path(window_path)
    source_root = Path(source_root)
    metadata = parse_real_recording_name(window_path.name)
    source_index = build_source_index(source_root) if source_index is None else source_index
    candidates = source_index.get(source_basename(window_path.name), [])
    start = metadata.segment_id * WINDOW_SAMPLES
    end = start + WINDOW_SAMPLES
    entry = {
        "schema_version": "1.0",
        "dataset_interpretation": "single_person_repeated_recordings",
        "person_id": PERSON_ID,
        "legacy_dataset_id": metadata.subject_id,
        "recording_date": metadata.date,
        "recording_time": metadata.time,
        "sensor_distance_cm": metadata.distance_cm,
        "recording_session_id": UNKNOWN,
        "continuous_source_id": UNKNOWN,
        "window_id": metadata.segment_id,
        "window_start_sample": start,
        "window_end_sample_exclusive": end,
        "source_filename": window_path.name,
        "source_recording_filename": source_basename(window_path.name),
        "source_relative_path": UNKNOWN,
        "file_sha256": sha256_file(window_path),
        "source_sha256": UNKNOWN,
        "data_origin": "physical_recording",
        "windowing_method": "contiguous_non_overlapping_1000_sample_slices",
        "overlap_status": UNKNOWN,
        "dependency_group": UNKNOWN,
        "allowed_split_group": UNKNOWN,
        "provenance_status": "UNKNOWN_SOURCE",
        "content_verified": False,
        "source_candidate_count": len(candidates),
        "matching_source_count": 0,
        "provenance_notes": "No unique continuous source file was found.",
    }
    if not candidates:
        return entry

    exact_matches = []
    if verify_content:
        window = np.load(window_path, mmap_mode="r")
        for candidate in candidates:
            source = np.load(candidate, mmap_mode="r")
            shape_ok = window.shape == (2, WINDOW_SAMPLES) and source.ndim == 2 and source.shape[0] >= 3
            range_ok = shape_ok and end <= source.shape[1]
            if range_ok and np.array_equal(window[0], source[2, start:end]) \
                    and np.array_equal(window[1], source[0, start:end]):
                exact_matches.append(candidate)
        entry["matching_source_count"] = len(exact_matches)
        if not exact_matches:
            entry["provenance_status"] = "CONTENT_MISMATCH"
            entry["provenance_notes"] = (
                f"{len(candidates)} basename candidate(s) found, but none matched the declared slice."
            )
            return entry
        source_path = exact_matches[0]
    elif len(candidates) == 1:
        source_path = candidates[0]
    else:
        entry["provenance_status"] = "AMBIGUOUS_SOURCE"
        entry["provenance_notes"] = f"{len(candidates)} matching source files require content verification."
        return entry

    relative = source_path.relative_to(source_root)
    source_digest = sha256_file(source_path)
    continuous_source_id = stable_source_id(source_digest)
    entry.update({
        "continuous_source_id": continuous_source_id,
        "source_relative_path": relative.as_posix(),
        "source_sha256": source_digest,
        "overlap_status": "KNOWN_NON_OVERLAPPING_WITHIN_SOURCE",
        "dependency_group": continuous_source_id,
        "allowed_split_group": continuous_source_id,
        "provenance_status": "SOURCE_MATCHED",
        "provenance_notes": (
            "Source matched by basename. Session ancestry above this continuous source is unknown."
        ),
    })
    if not verify_content:
        return entry
    entry["content_verified"] = True
    if len(exact_matches) == 1:
        entry["provenance_status"] = "VERIFIED_EXACT_SLICE"
        copy_note = ""
    else:
        entry["provenance_status"] = "VERIFIED_EXACT_SLICE_MULTIPLE_SOURCE_COPIES"
        copy_note = f" {len(exact_matches)} source paths contain the same matching slice; content identity defines the dependency group."
    entry["provenance_notes"] = (
        "BCG equals source channel 2 and ECG equals source channel 0 over the declared sample range; "
        "recording-session ancestry above this source remains unknown." + copy_note
    )
    return entry


def validate_manifest(entries):
    """Return machine-readable issues without assuming unknown sessions are independent."""
    issues = []
    checksums = defaultdict(list)
    sources = defaultdict(list)
    logical_ids = defaultdict(list)
    for row in entries:
        checksums[row["file_sha256"]].append(row)
        sources[row["continuous_source_id"]].append(row)
        logical_ids[(row["legacy_dataset_id"], row["recording_date"], row["recording_time"],
                     row["sensor_distance_cm"], row["window_id"])].append(row)
        if row["recording_session_id"] == UNKNOWN:
            issues.append({"severity": "warning", "code": "MISSING_RECORDING_SESSION",
                           "source_filename": row["source_filename"]})
        if not row["content_verified"]:
            issues.append({"severity": "error", "code": "UNVERIFIED_PROVENANCE",
                           "source_filename": row["source_filename"],
                           "status": row["provenance_status"]})
        if row.get("matching_source_count", 0) > 1:
            issues.append({"severity": "warning", "code": "MULTIPLE_MATCHING_SOURCE_COPIES",
                           "source_filename": row["source_filename"],
                           "matching_source_count": row["matching_source_count"]})

    for digest, rows in checksums.items():
        if len(rows) > 1:
            issues.append({"severity": "error", "code": "DUPLICATE_FILE_CHECKSUM",
                           "sha256": digest, "files": [row["source_filename"] for row in rows]})
    for logical_id, rows in logical_ids.items():
        if len(rows) > 1:
            issues.append({"severity": "error", "code": "DUPLICATE_LOGICAL_WINDOW_ID",
                           "logical_id": list(logical_id),
                           "files": [row["source_filename"] for row in rows]})
    for source_id, rows in sources.items():
        if source_id == UNKNOWN:
            continue
        split_groups = {row["allowed_split_group"] for row in rows}
        if len(split_groups) != 1:
            issues.append({"severity": "error", "code": "SOURCE_CROSSES_SPLIT_GROUPS",
                           "continuous_source_id": source_id, "split_groups": sorted(split_groups)})
        ordered = sorted(rows, key=lambda row: row["window_start_sample"])
        for previous, current in zip(ordered, ordered[1:]):
            if current["window_start_sample"] < previous["window_end_sample_exclusive"]:
                issues.append({"severity": "error", "code": "OVERLAPPING_WINDOWS",
                               "continuous_source_id": source_id,
                               "files": [previous["source_filename"], current["source_filename"]]})
            if (current["window_start_sample"] == previous["window_end_sample_exclusive"]
                    and current["allowed_split_group"] != previous["allowed_split_group"]):
                issues.append({"severity": "error", "code": "ADJACENT_WINDOWS_CROSS_SPLIT_GROUPS",
                               "continuous_source_id": source_id,
                               "files": [previous["source_filename"], current["source_filename"]]})

    counts = Counter(issue["severity"] for issue in issues)
    return {
        "entries": len(entries),
        "verified_exact_slices": sum(row["content_verified"] for row in entries),
        "unique_continuous_sources": len({row["continuous_source_id"] for row in entries
                                           if row["continuous_source_id"] != UNKNOWN}),
        "errors": counts["error"],
        "warnings": counts["warning"],
        "issues": issues,
    }
