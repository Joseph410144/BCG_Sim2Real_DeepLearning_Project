"""Filename metadata parsing for the current real-world BCG dataset."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path


REAL_DATA_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})_"
    r"(?P<distance>\d+)cm_Subject_(?P<subject>\d+)_heart_(?P<segment>\d+)\.npy$"
)


@dataclass(frozen=True)
class RecordingMetadata:
    filename: str
    date: str
    time: str
    distance_cm: int
    subject_id: int
    segment_id: int

    def to_dict(self):
        return asdict(self)


def parse_real_recording_name(path):
    filename = Path(path).name
    match = REAL_DATA_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized real-data filename: {filename}")
    values = match.groupdict()
    return RecordingMetadata(filename, values["date"], values["time"],
                             int(values["distance"]), int(values["subject"]),
                             int(values["segment"]))
