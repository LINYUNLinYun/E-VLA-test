"""Command-space Kalman filter; no EEF, labels or future chunks enter the state."""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)
SHAPES = ('isotropic', 'r_half', 'r_full', 'q_half', 'q_full', 'qr_half')
MAX_CONDITION = 1e12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_covariance(matrix, positive=True):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Invalid covariance array')
    if not np.allclose(a, a.T, rtol=0, atol=1e-10):
        raise ValueError('Asymmetric covariance')
    eigen = np.linalg.eigvalsh(a)
    invalid = eigen[0] <= 0 if positive else eigen[0] < -1e-10
    if invalid:
        raise ValueError('Non PD/PSD covariance')
    if positive and eigen[-1] / eigen[0] > MAX_CONDITION:
        raise ValueError('Unreasonable covariance condition number')
    return a


@dataclass(frozen=True)
class Parameters:
    q_u: float
    q_w: float
    g: float = 1.
    beta: float = 1.
    shape: str = 'isotropic'

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.q_u, self.q_w, self.g)):
            raise ValueError('Noise ratios and g must be finite positive')
        if not 0 < self.beta <= 1 or self.shape not in SHAPES:
            raise ValueError('Invalid decay/shape')


def candidate_manifest():
    return dict(schema=1, dt_s=.05, horizon=50, executed_prefix=5,
                stage1=[asdict(Parameters(q, w)) for q in GRID for w in GRID],
                stage2=dict(parents='top3 stage1 by frozen selection rule',
                            g=[.5, 1., 1.5], beta=[.7, .9, 1.], shapes=list(SHAPES)),
                max_candidates=211, covariance_eigen_floor=1e-6,
                covariance_recipe='R: first command differences; Q: second differences; episode boundaries excluded',
                shape_mix='half=0.5*I+0.5*normalized_empirical; full=normalized_empirical with eigen floor',
                fold='leave-one official init out; same init across suites',
                ranking='prefer no >5% tail regression in every fold; prefix equal mean, lateral p95, complexity, lexical',
                tail_reference='frozen old Kalman; both prefix L2 p95 and proxy lateral p95',
                reproduction_atol=1e-9, reproduction_rtol=1e-8,
                data_disclosure='Reused previously inspected v2 data, not a blind test')


def covariance_shape(differences, shrinkage):
    """Fit one trace-normalized PD shape. Zero variation explicitly yields identity."""
    if not 0 <= shrinkage <= 1:
        raise ValueError('Invalid shrinkage')
    a = np.asarray(differences, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Insufficient/invalid training differences')
    raw = np.cov(a, rowvar=False)
    raw = (raw + raw.T) / 2
    if np.trace(raw) <= 1e-15:
        raw = np.eye(3)
    else:
        raw *= 3 / np.trace(raw)
    eig, vectors = np.linalg.eigh(raw)
    raw = (vectors * np.maximum(eig, 1e-6)) @ vectors.T
    raw *= 3 / np.trace(raw)
    shape = (1 - shrinkage) * np.eye(3) + shrinkage * raw
    shape = (shape + shape.T) / 2
    shape *= 3 / np.trace(shape)
    check_covariance(shape)
    eigen = np.linalg.eigvalsh(shape)
    metadata = dict(matrix=shape.tolist(), eigenvalues=eigen.tolist(),
                    condition=float(eigen[-1]/eigen[0]), empirical_weight=shrinkage,
                    identity_shrinkage=1-shrinkage)
    metadata['sha256'] = digest(metadata)
    return shape, metadata


def fit_shapes(training, kind):
    """Only accept calibration episodes; caller supplies the current training fold."""
    if not training or any(key[1] != 0 for key in training):
        raise ValueError('Shape fitting requires task0 training episodes only')
    first, second = [], []
    for records in training.values():
        a = np.clip(np.asarray(records[0]['_actions'])[:, :3], -1, 1)
        first.extend(np.diff(a, axis=0))
        second.extend(np.diff(a, n=2, axis=0))
    qs = {'q_half': .5, 'q_full': 1., 'qr_half': .5}.get(kind, 0.)
    rs = {'r_half': .5, 'r_full': 1., 'qr_half': .5}.get(kind, 0.)
    q, qm = covariance_shape(second, qs)
    r, rm = covariance_shape(first, rs)
    return q, r, dict(q=qm, r=rm, training_episodes=[list(k) for k in sorted(training)])


class CommandKalman:
    def __init__(self, parameters, shape_q=None, shape_r=None):
        self.parameters = parameters
        q = check_covariance(np.eye(3) if shape_q is None else shape_q)
        r = check_covariance(np.eye(3) if shape_r is None else shape_r)
        if not np.isclose(np.trace(r)/3, 1., atol=1e-10):
            raise ValueError('R scale must be fixed at trace/3=1')
        eye, zero = np.eye(3), np.zeros((3, 3))
        self.F = np.block([[eye, eye], [zero, parameters.beta*eye]])
        self.F_forecast = np.block([[eye, parameters.g*eye], [zero, parameters.beta*eye]])
        self.H = np.concatenate((eye, zero), axis=1)
        self.Q = np.block([[parameters.q_u*q, zero], [zero, parameters.q_w*q]])
        self.R = r
        self.x = np.zeros(6)
        self.P = np.block([[r, zero], [zero, r]])
        self.innovation = None
        self.count = 0

    def observe(self, executed_action):
        a = np.asarray(executed_action, dtype=np.float64)
        if a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all():
            raise ValueError('Invalid executed action')
        z = np.clip(a[:3], -1, 1)
        x = self.F @ self.x
        p = self.F @ self.P @ self.F.T + self.Q
        innovation = z-self.H@x
        gain = np.linalg.solve(self.H@p@self.H.T+self.R, self.H@p).T
        self.x = x + gain@innovation
        joseph = np.eye(6)-gain@self.H
        self.P = joseph@p@joseph.T + gain@self.R@gain.T
        self.P = (self.P+self.P.T)/2
        check_covariance(self.P)
        if not np.isfinite(self.x).all():
            raise ValueError('Nonfinite posterior')
        self.innovation = innovation
        self.count += 1

    def forecast(self, horizon=50):
        if horizon < 1:
            raise ValueError('Positive horizon required')
        x, p = self.x.copy(), self.P.copy()
        means, covariances = [], []
        for _ in range(horizon):
            x = self.F_forecast@x
            p = self.F_forecast@p@self.F_forecast.T+self.Q
            p = (p+p.T)/2
            check_covariance(p)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite forecast')
            means.append(x[:3].copy())
            covariances.append(p[:3, :3].copy())
        return np.asarray(means), np.asarray(covariances)


def predict_episode(records, parameters, shape_q, shape_r, horizon=50):
    kalman = CommandKalman(parameters, shape_q, shape_r)
    last = 0
    for record in records:
        cutoff = int(record['environment_action_step'])
        if cutoff < last or cutoff > len(record['_actions']):
            raise ValueError('Invalid/nonmonotonic history cutoff')
        # The sole assimilation path. Full/variant chunks and telemetry are absent.
        for action in record['_actions'][last:cutoff]:
            kalman.observe(action)
        last = cutoff
        mean, cov = kalman.forecast(horizon)
        yield record, mean, cov, None if kalman.innovation is None else kalman.innovation.copy()
