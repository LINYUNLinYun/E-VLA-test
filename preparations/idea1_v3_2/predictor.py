"""Causal command-series extrapolators (not physical EEF trajectory models).

Input consists solely of already executed normalized XYZ commands. A fresh
instance is required per episode. Forecasting never mutates assimilation state.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    family: str
    window: int = 16
    order: int = 1
    decay: float = .9
    ridge: float = .1
    damping: float = .7


def candidates():
    out = []
    for window in (8, 16, 32):
        for order in (1, 2):
            for decay in (.7, .9):
                for damping in (.7, 1.):
                    out.append(Config('WLP', window, order, decay, .001, damping))
    for window in (5, 10, 20):
        for damping in (.7, .9, 1.):
            out.append(Config('ConstantTurn', window, 1, .9, .1, damping))
    for window in (16, 32):
        for order in (1, 2, 3):
            for ridge in (.01, .1, 1.):
                out.append(Config('VAR', window, order, .9, ridge, .9))
    return out


class Predictor:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.history = []

    def observe(self, action):
        a = np.asarray(action, dtype=float)
        if a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all():
            raise ValueError('Invalid executed command')
        self.history.append(np.clip(a[:3], -1, 1).copy())

    def forecast(self, horizon=50):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Positive integer horizon required')
        c = self.config
        a = np.asarray(self.history[-c.window:], dtype=float).reshape(-1, 3)
        if not len(a):
            return np.zeros((horizon, 3))
        if len(a) < max(4, c.order + 2):
            return np.tile(a[-1], (horizon, 1))
        if c.family == 'WLP':
            result = self._polynomial(a, horizon)
        elif c.family == 'ConstantTurn':
            result = self._turn(a, horizon)
        elif c.family == 'VAR':
            result = self._var(a, horizon)
        else:
            raise ValueError('Unknown predictor family')
        if result.shape != (horizon, 3) or not np.isfinite(result).all():
            raise ValueError('Unstable/nonfinite forecast')
        return np.clip(result, -1, 1)

    def _polynomial(self, a, horizon):
        c, n = self.config, len(a)
        # Normalize the time basis for conditioning; solve the weighted ridge
        # system, leaving the intercept unpenalized. Anchoring at last command
        # avoids an artificial jump from a smoothed fitted intercept.
        t = np.arange(1-n, 1, dtype=float)/(n-1)
        x = np.vander(t, c.order+1, increasing=True)
        w = c.decay ** np.arange(n-1, -1, -1)
        penalty = np.diag([0.] + [c.ridge]*c.order)
        coef = np.linalg.solve(x.T@(w[:, None]*x)+penalty, x.T@(w[:, None]*a))
        steps = np.cumsum(c.damping**np.arange(horizon)) / (n-1)
        future = np.vander(steps, c.order+1, increasing=True)@coef
        return a[-1] + future-coef[0]

    def _turn(self, a, horizon):
        c = self.config
        norm = np.linalg.norm(a, axis=1)
        valid = (norm[:-1] > .02) & (norm[1:] > .02)
        if valid.sum() < 2:
            return np.tile(a[-1], (horizon, 1))
        # Weighted Kabsch rotation between successive command directions.
        # This is a 3-D constant angular turn of command vectors; commands are
        # delta-position control targets, not Cartesian EEF position samples.
        left = a[:-1][valid]/norm[:-1][valid, None]
        right = a[1:][valid]/norm[1:][valid, None]
        w = c.decay**np.arange(len(a)-2, -1, -1)[valid]
        u, _, vt = np.linalg.svd(left.T@(w[:, None]*right))
        fix = np.diag([1., 1., np.linalg.det(u@vt)])
        rotation = u@fix@vt
        cosine = np.clip((np.trace(rotation)-1)/2, -1, 1)
        angle = min(float(np.arccos(cosine)), np.pi/4)
        axis = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.tile(a[-1], (horizon, 1))
        axis /= axis_norm
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        # Row-vector convention matches the Kabsch solve above.
        current, out = a[-1].copy(), []
        for h in range(horizon):
            theta = angle*c.damping**h
            r = np.eye(3)+np.sin(theta)*cross+(1-np.cos(theta))*(cross@cross)
            current = current@r
            out.append(current.copy())
        return np.asarray(out)

    def _var(self, a, horizon):
        c, p = self.config, self.config.order
        center = a.mean(axis=0)
        z = a-center
        x = np.asarray([z[t-p:t][::-1].reshape(-1) for t in range(p, len(a))])
        y = z[p:]
        # Ridge toward persistence, not toward zero velocity. Rescale design
        # by recent variation so ridge has a stable meaning across episodes.
        scale = max(float(np.mean(x*x)), 1e-5)
        prior = np.zeros((3*p, 3)); prior[:3] = np.eye(3)
        lam = c.ridge*len(x)*scale
        coef = np.linalg.solve(x.T@x+lam*np.eye(3*p), x.T@y+lam*prior)
        companion = np.zeros((3*p, 3*p)); companion[:3] = coef.T
        if p > 1:
            companion[3:, :-3] = np.eye(3*(p-1))
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
        # Shrink the learned update toward HoldLast when the local fit is
        # explosive. Every rollout is additionally bounded to command range.
        if radius > .995:
            coef /= max(radius/.995, 1.)**p
        seq, out = list(z), []
        for h in range(horizon):
            value = np.asarray(seq[-p:][::-1]).reshape(-1)@coef
            value = np.clip(value+center, -1, 1)-center
            # Damped transition reduces unsupported long extrapolation.
            value = seq[-1] + c.damping**h*(value-seq[-1])
            seq.append(value); out.append(value+center)
        return np.asarray(out)


def predict_history(actions, cutoff, config, horizon=50):
    if cutoff < 0 or cutoff > len(actions):
        raise ValueError('History cutoff outside executed timeline')
    model = Predictor(config)
    for action in actions[max(0, cutoff-config.window):cutoff]:
        model.observe(action)
    return model.forecast(horizon)
