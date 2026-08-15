import unittest

import numpy as np

from Algorithm.confidence import BinnedIsotonicCalibrator
from calibrate_confidence import binary_auc


class ConfidenceTests(unittest.TestCase):
    def test_calibrator_is_monotonic_and_bounded(self):
        scores = np.linspace(0, 1, 100)
        labels = np.asarray([0] * 30 + [1] * 20 + [0] * 10 + [1] * 40)
        model = BinnedIsotonicCalibrator(10).fit(scores, labels)
        predictions = model.predict(scores)
        self.assertTrue(np.all(np.diff(predictions) >= -1e-12))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

    def test_auc(self):
        self.assertEqual(binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
