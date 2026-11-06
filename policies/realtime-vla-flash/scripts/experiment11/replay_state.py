"""Recover the original client state from lossless real telemetry, not video."""
import math
import numpy as np


def client_state_from_telemetry(row, cached_state):
    """Use the exact client's quaternion conversion; require cache round-trip."""
    quaternion = np.asarray(row['eef_quat'], dtype=np.float64).copy()
    if quaternion.shape != (4,):
        raise RuntimeError('Invalid telemetry quaternion')
    quaternion[3] = np.clip(quaternion[3], -1., 1.)
    denominator = np.sqrt(1.-quaternion[3]*quaternion[3])
    rotation = (np.zeros(3) if math.isclose(denominator, 0.) else
                quaternion[:3]*2.*math.acos(quaternion[3])/denominator)
    state = np.concatenate((np.asarray(row['eef_pos'], dtype=np.float64), rotation,
                            np.asarray(row['gripper_qpos'], dtype=np.float64)))
    cached = np.asarray(cached_state)
    if state.shape != (8,) or cached.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError('Invalid telemetry/client state')
    if not np.array_equal(state.astype(cached.dtype), cached):
        raise RuntimeError('Telemetry state does not round-trip to frozen input')
    return state
