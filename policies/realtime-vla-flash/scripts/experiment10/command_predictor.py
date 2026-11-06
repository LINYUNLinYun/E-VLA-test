"""Executed-command Kalman predictor and conservative contiguous-prefix gate.

This is not robot dynamics, visual verification, or proof of task correctness.
Thresholds must be calibrated on validation tasks before online evaluation.
"""
import numpy as np


class CommandPredictor:
    def __init__(self, process_variance, measurement_variance, minimum_history=4):
        self.q = np.asarray(process_variance, dtype=float)
        self.r = np.asarray(measurement_variance, dtype=float)
        if self.q.shape != (6,) or self.r.shape != (6,) or not np.isfinite(self.q).all() or not np.isfinite(self.r).all():
            raise ValueError('Six finite command variances required')
        if np.any(self.q <= 0) or np.any(self.r <= 0) or minimum_history < 2:
            raise ValueError('Variances must be positive; history at least two')
        self.minimum_history = minimum_history
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None
        self.history_count = 0
        self.last_gripper = None

    def observe_executed(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('Only finite actually executed [N,7] commands may update the filter')
        transition = np.array([[1., 1.], [0., 1.]])
        for action in actions:
            if self.state is None:
                self.state = np.column_stack((action[:6], np.zeros(6)))
                self.covariance = np.stack([np.diag([v, v]) for v in self.r])
            else:
                prediction = self.state @ transition.T
                covariance = transition @ self.covariance @ transition.T + self.q[:, None, None]*np.eye(2)
                gain = covariance[:, :, 0]/(covariance[:, 0, 0]+self.r)[:, None]
                self.state = prediction+gain*(action[:6]-prediction[:, 0])[:, None]
                # Joseph form preserves PSD under numerical roundoff.
                left = np.broadcast_to(np.eye(2), covariance.shape).copy()
                left[:, :, 0] -= gain
                self.covariance = left @ covariance @ left.transpose(0, 2, 1) + self.r[:, None, None]*gain[:, :, None]*gain[:, None, :]
            self.last_gripper = bool(action[6] > 0)
            self.history_count += 1

    def predict(self, length):
        if self.history_count < self.minimum_history or length < 1:
            raise ValueError('Insufficient executed history or invalid horizon')
        state, covariance = self.state.copy(), self.covariance.copy()
        transition = np.array([[1., 1.], [0., 1.]])
        means, variances = [], []
        for _ in range(length):
            state = state @ transition.T
            covariance = transition @ covariance @ transition.T + self.q[:, None, None]*np.eye(2)
            means.append(state[:, 0].copy())
            variances.append(covariance[:, 0, 0]+self.r)
        return np.asarray(means), np.asarray(variances)


def accepted_prefix(predictor, candidate, translation_threshold, rotation_threshold,
                    observation_age_ms, max_observation_age_ms, maximum_horizon=12):
    candidate = np.asarray(candidate, dtype=float)
    if (candidate.ndim != 2 or candidate.shape[1] != 7 or not np.isfinite(candidate).all() or
        predictor.history_count < predictor.minimum_history):
        return 0, 'invalid_candidate_or_insufficient_history'
    if (not np.isfinite(observation_age_ms) or observation_age_ms < 0 or observation_age_ms > max_observation_age_ms):
        return 0, 'observation_too_old_or_invalid'
    if any(not np.isfinite(t) or t <= 0 for t in (translation_threshold, rotation_threshold, max_observation_age_ms)):
        raise ValueError('Frozen positive thresholds required')
    length = min(len(candidate), maximum_horizon)
    if length < 1:
        return 0, 'empty_prefix'
    means, variances = predictor.predict(length)
    normalized = (candidate[:length, :6]-means)**2/variances
    for index in range(length):
        if bool(candidate[index, 6] > 0) != predictor.last_gripper:
            return index, 'gripper_switch'
        if normalized[index, :3].sum() > translation_threshold or normalized[index, 3:].sum() > rotation_threshold:
            return index, 'command_prediction_disagreement'
    return length, 'within_prediction_thresholds_not_semantically_verified'


def verified_prefix(candidate, reference, maximum_horizon,
                    translation_threshold_m, rotation_threshold_rad):
    """Return the longest contiguous candidate prefix matching a same-observation reference."""
    candidate, reference = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if (candidate.ndim != 2 or candidate.shape[1] != 7 or reference.shape != candidate.shape or
        not np.isfinite(candidate).all() or not np.isfinite(reference).all() or maximum_horizon < 1):
        return 0, 'invalid_candidate_or_reference'
    if (not np.isfinite(translation_threshold_m) or translation_threshold_m <= 0 or
        not np.isfinite(rotation_threshold_rad) or rotation_threshold_rad <= 0):
        raise ValueError('Frozen positive translation and rotation thresholds required')
    length = min(len(candidate), int(maximum_horizon))
    for index in range(length):
        if bool(candidate[index, 6] > 0) != bool(reference[index, 6] > 0):
            return index, 'gripper_disagreement'
        if np.linalg.norm(candidate[index, :3] - reference[index, :3]) > translation_threshold_m:
            return index, 'translation_disagreement'
        if np.linalg.norm(candidate[index, 3:6] - reference[index, 3:6]) > rotation_threshold_rad:
            return index, 'rotation_disagreement'
    return length, 'within_reference_thresholds_not_semantically_verified'
