"""Label-free affine alignment from streaming paired-state statistics.

Arrays contain one token per row. ``x @ weight + bias`` maps Source states to
Target states. No semantic labels are accepted by the alignment interface.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Affine:
    weight: np.ndarray
    bias: np.ndarray

    def __call__(self, x):
        return np.asarray(x, dtype=self.weight.dtype) @ self.weight + self.bias


class PairedStatistics:
    def __init__(self, source_width, target_width):
        self.n = 0
        self.sx = np.zeros(source_width, np.float64)
        self.sy = np.zeros(target_width, np.float64)
        self.xx = np.zeros((source_width, source_width), np.float64)
        self.xy = np.zeros((source_width, target_width), np.float64)
        self.y2 = 0.

    def add(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not len(x):
            raise ValueError("Provide nonempty paired token matrices")
        if x.shape[1] != len(self.sx) or y.shape[1] != len(self.sy):
            raise ValueError("Representation widths changed")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite training states")
        self.n += len(x)
        self.sx += x.sum(axis=0)
        self.sy += y.sum(axis=0)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.y2 += float(np.square(y).sum())

    def centered(self):
        if self.n < 2:
            raise ValueError("At least two paired training tokens are required")
        mx, my = self.sx / self.n, self.sy / self.n
        xx = self.xx - self.n * np.outer(mx, mx)
        xy = self.xy - self.n * np.outer(mx, my)
        return mx, my, (xx + xx.T) / 2, xy

    def fit(self, relative_ridge):
        if not np.isfinite(relative_ridge) or relative_ridge <= 0:
            raise ValueError("Positive finite ridge regularization is required")
        mx, my, xx, xy = self.centered()
        scale = float(np.trace(xx) / len(mx))
        if scale <= 0:
            raise ValueError("Source states have no centered variance")
        weight = np.linalg.solve(xx + np.eye(len(mx)) * relative_ridge * scale, xy)
        return Affine(weight.astype(np.float32), (my - mx @ weight).astype(np.float32))

    def mean_scale(self):
        mx, my, xx, _ = self.centered()
        if len(mx) != len(my):
            raise ValueError("Mean/scale control requires matching widths")
        vx = float(np.trace(xx))
        vy = self.y2 - self.n * float(np.square(my).sum())
        if vx <= 0 or vy < 0:
            raise ValueError("Invalid centered variance")
        scale = np.sqrt(vy / vx)
        return Affine(np.eye(len(mx), dtype=np.float32) * scale, (my - mx * scale).astype(np.float32))


def mean_squared_error(mapping, batches):
    total, count = 0., 0
    for x, y in batches:
        error = mapping(x).astype(np.float64) - np.asarray(y, np.float64)
        total += float(np.square(error).sum())
        count += error.size
    if not count:
        raise ValueError("Empty alignment validation split")
    return total / count
