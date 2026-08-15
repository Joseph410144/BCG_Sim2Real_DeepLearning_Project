"""Small dependency-free monotonic confidence calibrator."""

import numpy as np


class BinnedIsotonicCalibrator:
    def __init__(self, bins=20):
        self.bins = bins
        self.x_ = None
        self.y_ = None

    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        mask = np.isfinite(scores) & np.isfinite(labels)
        scores, labels = scores[mask], labels[mask]
        if not len(scores):
            raise ValueError("No finite calibration samples")
        order = np.argsort(scores)
        groups = np.array_split(order, min(self.bins, len(order)))
        blocks = [[float(np.mean(scores[group])), float(np.mean(labels[group])), len(group)] for group in groups]
        index = 0
        while index < len(blocks) - 1:
            if blocks[index][1] <= blocks[index + 1][1]:
                index += 1
                continue
            left, right = blocks[index], blocks[index + 1]
            weight = left[2] + right[2]
            merged = [
                (left[0] * left[2] + right[0] * right[2]) / weight,
                (left[1] * left[2] + right[1] * right[2]) / weight,
                weight,
            ]
            blocks[index:index + 2] = [merged]
            index = max(0, index - 1)
        self.x_ = np.asarray([block[0] for block in blocks])
        self.y_ = np.asarray([block[1] for block in blocks])
        return self

    def predict(self, scores):
        if self.x_ is None:
            raise RuntimeError("Calibrator must be fitted before prediction")
        scores = np.asarray(scores, dtype=float)
        return np.clip(np.interp(scores, self.x_, self.y_, left=self.y_[0], right=self.y_[-1]), 0, 1)
