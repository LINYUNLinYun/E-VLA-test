"""Post hoc command/proxy diagnostics; these values never enter the filter."""
import numpy as np

PHASES = {'initial': 'initial/approach', 'approach': 'initial/approach',
          'contact': 'contact/grasp', 'grasp_or_manipulate': 'contact/grasp',
          'transport': 'transport', 'goal_alignment': 'alignment/release',
          'release_or_final_action': 'alignment/release'}


def decompose(predicted, reference):
    a = np.clip(np.asarray(predicted, dtype=float)[:, :3], -1, 1)*50
    b = np.clip(np.asarray(reference, dtype=float)[:, :3], -1, 1)*50
    if a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError('Invalid paired command blocks')
    error = a-b
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    directional = (na >= 1) & (nb >= 1)
    cosine = np.full(len(a), np.nan)
    cosine[directional] = np.clip(np.sum(a[directional]*b[directional], axis=1) /
                                  (na[directional]*nb[directional]), -1, 1)
    # Local command step is the Full proxy tangent. Reversals and abrupt turns
    # (>90 degrees between adjacent defined steps) have unstable tangents.
    tangent_valid = nb >= 1
    unstable = np.zeros(len(a), dtype=bool)
    if len(a) > 1:
        unstable[1:] = (np.sum(b[1:]*b[:-1], axis=1) <= 0) & (nb[1:] >= 1) & (nb[:-1] >= 1)
    tangent_valid &= ~unstable
    tangent = np.divide(b, nb[:, None], out=np.zeros_like(b), where=nb[:, None] > 0)
    delta = np.cumsum(error, axis=0)
    along = np.sum(delta*tangent, axis=1)
    lateral = np.linalg.norm(delta-along[:, None]*tangent, axis=1)
    along[~tangent_valid], lateral[~tangent_valid] = np.nan, np.nan
    ratio = np.divide(na, nb, out=np.full_like(na, np.nan), where=nb >= 1)
    return dict(error_mm=error, abs_error_mm=np.abs(error), positive=error > 0,
                negative=error < 0, l2_mm=np.linalg.norm(error, axis=1),
                magnitude_difference_mm=na-nb, magnitude_ratio=ratio,
                cosine=cosine, angle_rad=np.arccos(cosine), direction_valid=directional,
                proxy_delta_mm=delta, along_mm=along, lateral_mm=lateral,
                tangent_valid=tangent_valid, reference_low_motion=nb < 1,
                reference_cumulative_distance_mm=np.cumsum(nb),
                predicted_cumulative_distance_mm=np.cumsum(na))


def phase_shift_oracle(predicted, reference, window=12, radius=3, tolerance=1e-9):
    """Compare identical Full indices 4..9 at all shifts for default window=12.

    Positive delta compares predicted[index+delta] with Full[index]. Every shift
    uses window-2*radius samples. This is a post hoc oracle, never a correction.
    """
    a, b = np.asarray(predicted)[:, :3], np.asarray(reference)[:, :3]
    if window <= 2*radius or min(len(a), len(b)) < window:
        raise ValueError('Insufficient fixed matching window')
    a, b = np.clip(a, -1, 1)*50, np.clip(b, -1, 1)*50
    indices = np.arange(radius, window-radius)
    shifts = np.arange(-radius, radius+1)
    errors = np.array([np.linalg.norm(a[indices+d]-b[indices], axis=1).mean() for d in shifts])
    minimum = errors.min()
    tied = shifts[np.abs(errors-minimum) <= tolerance]
    best = min(tied, key=lambda d: (abs(d), d))
    residual = a[indices+best]-b[indices]
    norms = np.linalg.norm(b[indices], axis=1)
    valid = norms >= 1
    tangents = np.divide(b[indices], norms[:, None], out=np.zeros_like(residual), where=norms[:, None] > 0)
    lateral = np.linalg.norm(residual-np.sum(residual*tangents, axis=1)[:, None]*tangents, axis=1)
    return dict(oracle=True, positive_delta='predicted later offset matched to Full current offset',
                reference_offsets=(indices+1).tolist(), shifts=shifts.tolist(), errors_mm=errors.tolist(),
                zero_error_mm=float(errors[radius]), best_shift=int(best),
                improvement_mm=float(errors[radius]-minimum), residual_mm=float(minimum),
                residual_lateral_mm=float(lateral[valid].mean()) if valid.any() else None,
                unidentifiable=len(tied) > 1, tied_shifts=tied.tolist())


def geometry(telemetry, cutoff, window=5, threshold=.95):
    if cutoff < window:
        return dict(group='cold_start', straightness=None, speed_m_s=None, angles_rad=[])
    points = np.asarray([r['eef_pos'] for r in telemetry[cutoff-window:cutoff+1]], dtype=float)
    if points.shape != (window+1, 3) or not np.isfinite(points).all():
        raise ValueError('Missing/invalid EEF history')
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    distance = lengths.sum()
    straightness = float(np.linalg.norm(points[-1]-points[0])/distance) if distance > 0 else None
    angles = []
    for left, right, nl, nr in zip(steps[:-1], steps[1:], lengths[:-1], lengths[1:]):
        angles.append(float(np.arccos(np.clip(left@right/(nl*nr), -1, 1))) if nl > 1e-10 and nr > 1e-10 else None)
    return dict(group='low_motion' if distance < .020 else 'near_straight' if straightness >= threshold else 'curved',
                straightness=straightness, path_length_mm=float(distance*1000),
                speed_m_s=float(distance/(window*.05)), angles_rad=angles)


def scene_annotations(record):
    telemetry, t = record['_telemetry'], int(record['environment_action_step'])
    phases = [PHASES.get(r.get('semantic_phase'), 'unknown') for r in telemetry]
    switches = [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
    return dict(phase=PHASES.get(record.get('semantic_phase'), 'unknown'),
                transition=any(abs(t-i) <= 5 for i in switches),
                transition_is_posthoc=True, geometry=geometry(telemetry, t),
                sensitivity={f'w{w}_threshold{threshold}': geometry(telemetry, t, w, threshold)
                             for w in (5, 10) for threshold in (.90, .95, .98)})
