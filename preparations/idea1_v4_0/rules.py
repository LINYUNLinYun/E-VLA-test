"""CPU-only, causal phase routing and the existing frozen Kalman predictor."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def motion_features(commands, quaternions):
    if len(commands) < 5 or len(quaternions) < 5:
        return dict(ready=False, risk=0., gripper=False)
    a = np.asarray(commands[-5:], dtype=np.float64)
    q = np.asarray(quaternions[-5:], dtype=np.float64)
    if a.shape != (5, 7) or q.shape != (5, 4) or not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Five finite executed commands and world xyzw poses required')
    norms = np.linalg.norm(a[:, :3], axis=1)
    turns = []
    for i in range(4):
        if max(norms[i:i+2]) < 1e-4:
            turns.append(0.)
        elif min(norms[i:i+2]) < 1e-4:
            turns.append(180.)
        else:
            cosine = np.dot(a[i, :3], a[i+1, :3]) / (norms[i] * norms[i+1])
            turns.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    qnorm = np.linalg.norm(q, axis=1)
    if np.any(qnorm < 1e-8):
        raise ValueError('Invalid executed quaternion')
    q = q / qnorm[:, None]
    pose = float(np.degrees(2*np.arccos(np.clip(np.abs(q @ q.T), 0, 1))).max())
    change = float(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).max())
    rotation = float(np.linalg.norm(np.diff(a[:, 3:6], axis=0), axis=1).max())
    # Same command/pose quantities and scales as the previous decision machine.
    risk = max(max(turns)/20., change/.05, pose/5., rotation/.05)
    return dict(ready=True, risk=float(risk), gripper=bool(np.any(np.diff(a[:, 6] < 0))),
                turn_deg=max(turns), translation_change=change, pose_deg=pose,
                rotation_change=rotation, translation_norm=float(norms.mean()))


def skip_budget(features, thresholds):
    if not features['ready']:
        return 6, 'history_warmup'
    if features['gripper']:
        return 5, 'gripper_precision'
    score = features['risk']
    if score <= thresholds['q25']:
        return 8, 'very_smooth'
    if score <= thresholds['q50']:
        return 7, 'smooth'
    if score <= thresholds['q85']:
        return 6, 'general'
    return 5, 'precision_proxy'


class KalmanPrior:
    """Only acknowledged controller commands enter the filter.

    Reuses the previous frozen xyz command filter. Rotation/gripper hold their
    last executed values; all 25 unused raw action dimensions are zero.
    """
    def __init__(self):
        import sys
        root = str(ROOT/'policies/realtime-vla-flash')
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.idea1_v2_1.filter import Parameters, CommandKalman
        frozen = json.loads((ROOT/'preparations/idea1_v3_3/kalman_frozen.json').read_text())
        self.filter = CommandKalman(Parameters(**frozen['parameters']), frozen['shape_q'], frozen['shape_r'])
        self.last = None

    def observe(self, command):
        a = np.asarray(command, dtype=np.float32).copy()
        a[:6] = np.clip(a[:6], -1, 1)
        self.filter.observe(a)
        self.last = a

    def forecast(self):
        if self.last is None:
            raise RuntimeError('Kalman prior requires acknowledged actions')
        xyz, _ = self.filter.forecast(50)
        result = np.tile(self.last, (50, 1))
        result[:, :3] = np.clip(xyz, -1, 1)
        return result.astype(np.float32)
