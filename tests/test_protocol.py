import unittest

import numpy as np
import pandas as pd

from evaluate_heart_v7_protocol import fusion_predictions, subject_macro_mae


class ProtocolTests(unittest.TestCase):
    def test_fusion_switches_only_on_configured_conflict(self):
        data = pd.DataFrame({
            "heart_v6_hr": [140.0, 50.0, 70.0],
            "autocorrelation_hr": [70.0, 70.0, 71.0],
            "heart_v6_confidence": [0.1, 0.1, 0.9],
            "autocorrelation_confidence": [0.9, 0.9, 0.9],
        })
        parameters = {"high_v6_conf_max": 0.35, "high_acf_conf_min": 0.8,
                      "low_v6_conf_max": 0.35, "low_acf_conf_min": 0.2}
        predictions, high, low = fusion_predictions(data, parameters)
        np.testing.assert_allclose(predictions, [70, 70, 70])
        np.testing.assert_array_equal(high, [True, False, False])
        np.testing.assert_array_equal(low, [False, True, False])

    def test_subject_macro_mae_weights_subjects_equally(self):
        data = pd.DataFrame({"subject_id": [1, 1, 1, 2], "ecg_hr": [60, 60, 60, 60]})
        self.assertEqual(subject_macro_mae(data, np.array([60, 60, 60, 70])), 5.0)


if __name__ == "__main__":
    unittest.main()
