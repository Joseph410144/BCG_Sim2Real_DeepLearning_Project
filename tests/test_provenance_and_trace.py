import tempfile
import unittest
from pathlib import Path

import numpy as np

from Algorithm.heart_rate import heart_v6
from Algorithm.heart_v6_trace import trace_heart_v6
from Dataset.provenance import build_source_index, manifest_entry, validate_manifest
from run_controlled_am_experiment import low_frequency_metrics, transformed_signals


class ProvenanceAndTraceTests(unittest.TestCase):
    def test_manifest_resolves_exact_source_slice_among_same_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows, sources = root / "windows", root / "sources"
            windows.mkdir(); (sources / "a").mkdir(parents=True); (sources / "b").mkdir(parents=True)
            basename = "2025-01-01_1200_28cm_Subject_2_heart.npy"
            rng = np.random.default_rng(3)
            source = rng.normal(size=(3, 2000))
            other = rng.normal(size=(3, 2000))
            np.save(sources / "a" / basename, other)
            np.save(sources / "b" / basename, source)
            window_name = "2025-01-01_1200_28cm_Subject_2_heart_1.npy"
            np.save(windows / window_name, np.asarray([source[2, 1000:2000], source[0, 1000:2000]]))
            row = manifest_entry(windows / window_name, sources, build_source_index(sources))
            self.assertTrue(row["content_verified"])
            self.assertEqual(row["source_candidate_count"], 2)
            self.assertEqual(row["matching_source_count"], 1)
            self.assertEqual(row["window_start_sample"], 1000)
            self.assertEqual(row["allowed_split_group"], row["continuous_source_id"])

    def test_manifest_validation_detects_duplicate_window_content(self):
        base = {
            "source_filename": "a.npy", "file_sha256": "x", "legacy_dataset_id": 1,
            "recording_date": "2025-01-01", "recording_time": "1200",
            "sensor_distance_cm": 28, "window_id": 0, "recording_session_id": "UNKNOWN",
            "continuous_source_id": "source_a", "allowed_split_group": "source_a",
            "window_start_sample": 0, "window_end_sample_exclusive": 1000,
            "content_verified": True, "matching_source_count": 1,
            "provenance_status": "VERIFIED_EXACT_SLICE",
        }
        other = {**base, "source_filename": "b.npy", "window_id": 1,
                 "window_start_sample": 1000, "window_end_sample_exclusive": 2000}
        report = validate_manifest([base, other])
        self.assertTrue(any(issue["code"] == "DUPLICATE_FILE_CHECKSUM" for issue in report["issues"]))

    def test_trace_matches_production_heart_v6(self):
        fs = 100
        time = np.arange(1000) / fs
        values = (1 + 0.7 * np.sin(2 * np.pi * 1.2 * time)) * np.sin(2 * np.pi * 5 * time)
        production = heart_v6(values, fs=fs)
        trace = trace_heart_v6(values, fs=fs)
        self.assertEqual(trace.result.bpm, production.bpm)
        self.assertEqual(trace.band_signals.shape, (6, 1000))
        self.assertEqual(trace.rectified_bands.shape, (6, 1000))
        self.assertEqual(len(trace.peaks) - 1, len(trace.intervals_seconds))

    def test_am_envelope_recovers_known_modulation(self):
        fs = 100
        time = np.arange(1000) / fs
        values = (1 + 0.5 * np.cos(2 * np.pi * 1.2 * time)) * np.cos(2 * np.pi * 6 * time)
        methods, _ = transformed_signals(values, fs, carrier_hz=6)
        prediction, _, recovered, _, _ = low_frequency_metrics(
            methods["full_wave_rectified"], fs, 1.2, (0.5, 3.0), 0.1, 0.15)
        self.assertTrue(recovered)
        self.assertAlmostEqual(prediction, 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
