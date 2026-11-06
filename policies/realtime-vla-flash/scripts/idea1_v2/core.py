from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np

SUITES = ('libero_object', 'libero_spatial', 'libero_goal', 'libero_10')
LIMITS = dict(zip(SUITES, (280, 220, 300, 520)))
WEIGHT = 'c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891'
SKIP = (3, 4, 9, 11)
GPU_DISCOVERY_INTERVAL_SECONDS = 30
REPO = Path(__file__).resolve().parents[2]
OLD = REPO / 'results/experiment11_layer_skip_prefix_20261010'
NEW = REPO / 'results/idea1_v2_pi_alternation_kalman_20261014'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects frozen scientific configuration and completed outputs.
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        return
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)

def output_root(path):
    path = Path(path).resolve()
    if NEW.resolve() not in path.parents:
        raise ValueError('Results must use a distinct attempt under %s' % NEW)
    return path

def noise(init, replan):
    return np.random.default_rng(np.random.SeedSequence([12345, init, replan])).standard_normal((50,32)).astype(np.float32)

class Router:
    """Current/history-only routing. Call once per unique replan, not per action."""
    def __init__(self, phase=False):
        self.phase = phase
        self.reset()

    def reset(self):
        self.slot = 0
        self.latch = False
        self.initial_progress = None
        self.last_step = -1
        self.history = []

    @staticmethod
    def valid(state):
        names = state.get('primary_objects', [])
        if not names or state.get('goal_progress') is None:
            return False
        for name in names:
            # Fixtures need a verified surface mapping; center distances are insufficient.
            if state.get('objects', {}).get(name, {}).get('type') != 'object':
                return False
            for key in ('eef_object_distances', 'object_displacements'):
                value = state.get(key, {}).get(name)
                if value is None or not np.isfinite(value):
                    return False
            for key in ('robot_contacts', 'robot_grasps'):
                if not isinstance(state.get(key, {}).get(name),bool):
                    return False
        return np.isfinite(state['goal_progress'])

    def choose(self, states=()):
        if not self.phase:
            slot = self.slot
            self.slot = (slot+1)%3
            return ('V14' if slot < 2 else 'Full'), 'fixed', slot
        for state in states:
            step = int(state['step'])
            if step <= self.last_step:
                continue
            if self.last_step >= 0 and step != self.last_step+1:
                raise ValueError('Online state history has a gap')
            self.last_step = step
            self.history.append(state)
            if self.initial_progress is None:
                self.initial_progress = state.get('goal_progress')
            # A positive observed grasp remains meaningful even when another core
            # field is missing; it permanently protects the rest of the episode.
            self.latch |= any(state.get('robot_grasps',{}).values())
            if self.valid(state):
                self.latch |= (any(state['robot_grasps'].values()) or
                               max(state['object_displacements'].values()) >= .03 or
                               (self.initial_progress is not None and state['goal_progress'] > self.initial_progress))
        recent = self.history[-6:]
        current = recent[-1] if recent else {}
        reason = 'unknown'
        free = False
        if self.valid(current):
            if self.latch:
                reason = 'precision_latch'
            elif any(any(s.get('robot_contacts', {}).values()) for s in recent) or any(current['robot_grasps'].values()):
                reason = 'contact_or_grasp'
            elif min(current['eef_object_distances'].values()) <= .08:
                reason = 'precision_distance'
            elif len(self.history) < 3:
                reason = 'insufficient_history'
            else:
                free = all(self.valid(s) and min(s['eef_object_distances'].values()) > .08 and
                           not any(s['robot_contacts'].values()) and not any(s['robot_grasps'].values()) for s in self.history[-3:])
                reason = 'free_approach' if free else 'history_guard'
        if not free:
            self.slot = 0
            return 'Full', reason, None
        slot = self.slot
        self.slot = (slot+1)%3
        return ('V14' if slot < 2 else 'Full'), reason, slot

class Kalman:
    def __init__(self, q0, r0, dt=.05):
        self.dt = dt
        I = np.eye(3)
        self.F = np.block([[I,dt*I],[np.zeros((3,3)),I]])
        self.H = np.concatenate([I,np.zeros((3,3))],axis=1)
        self.Q = q0*np.block([[I/3,I/(2*dt)],[I/(2*dt),I/(dt*dt)]])
        self.R = r0*I
        self.x = np.zeros(6)
        self.P = np.diag([1]*3+[1/dt**2]*3)
        self.count = 0

    def observe(self, action):
        z = np.clip(np.asarray(action,dtype=np.float64)[:3],-1,1)
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite executed action')
        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T+self.Q
        K = np.linalg.solve(self.H@self.P@self.H.T+self.R,self.H@self.P).T
        self.x += K@(z-self.H@self.x)
        A = np.eye(6)-K@self.H
        self.P = A@self.P@A.T+K@self.R@K.T
        self.P = (self.P+self.P.T)/2
        self.count += 1

    def forecast(self, horizon=50):
        x,P = self.x.copy(),self.P.copy()
        means,cov = [],[]
        for _ in range(horizon):
            x = self.F@x
            P = self.F@P@self.F.T+self.Q
            means.append(x[:3].copy())
            cov.append(P[:3,:3].copy())
        return np.asarray(means),np.asarray(cov)
