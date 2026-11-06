"""Preparation only: periodic VLM/AE paths, including a Full-VLM FastD2 path.

The caller must supply a frozen condition validator before use. No scientific
condition whitelist or active rollout is defined by this preparation.
"""
import hashlib
import time
import numpy as np


class Policy:
    def __init__(self, backend, validator=None):
        if validator is None:
            from .common import validate_condition
            validator = validate_condition
        self.validator = validator
        self.b = backend
        self.config = None
        self.reset()

    def reset(self):
        self.pending = None
        self.round = self.step = self.draft_rounds = 0
        self.history = []
        self.counts = dict(full=0, draft=0, verify=0)

    def configure(self, config):
        self.mask, self.ae_mask = self.validator(config)
        self.config = dict(config)
        self.reset()
        return {}

    def predict(self, obs, instruction, suite, seed):
        if self.config is None:
            raise ValueError("Configure before prediction")
        if self.pending is not None:
            raise RuntimeError("Unacknowledged execution")
        tick = time.perf_counter()
        cfg = self.config
        full = cfg["mode"] == "full" or not self.history or self.draft_rounds >= cfg["period"]
        age = None if not self.history else self.step - self.history[-1]["step"]
        gap = None if len(self.history) < 2 else self.history[-1]["step"] - self.history[-2]["step"]
        ratio = 0. if gap is None else age / gap
        if full:
            c, timing = self.b.prepare_residual(obs, instruction, capture_layers=self.mask)
            actions, raw, cost = self.b.sample(c, seed, cfg["refresh_D"], ae_skip=())
            if cfg["mode"] != "full":
                self.history.append(dict(step=self.step, residuals=c["residuals"]))
                self.history = self.history[-2:]
            self.draft_rounds = 0
            self.counts["full"] += 1
        else:
            c, timing = self.b.prepare_residual(obs, instruction, skip=self.mask,
                history=self.history[-1]["residuals"],
                older=None if len(self.history) < 2 else self.history[-2]["residuals"], ratio=ratio)
            actions, raw, cost = self.b.sample(c, seed, cfg["D"], ae_skip=self.ae_mask)
            self.draft_rounds += 1
            self.counts["draft"] += 1
        actions = np.asarray(actions, np.float32)
        raw = raw.float().cpu().numpy()
        if not np.isfinite(actions).all() or not np.isfinite(raw).all():
            raise ValueError("Nonfinite actions")
        timing.update(cost)
        timing["policy_wall_ms"] = (time.perf_counter() - tick) * 1000
        noise = np.random.default_rng(seed).standard_normal((50, 32), dtype=np.float32)
        result = dict(actions=actions, raw=raw, arrays={}, execute_length=cfg["R"], timing=timing,
            seed=seed, round=self.round, step=self.step, counts=dict(self.counts), full_cache_age=age,
            route=dict(path="full" if full else "extrapolate", refresh=full, skip_layers=[] if full else list(self.mask),
                capture_layers=list(self.mask) if full else [], compensation=cfg["compensation"],
                ae_bypassed_layers=[] if full else list(self.ae_mask),
                ae_execution_audit=cost["ae_execution_audit"],
                has_second_history=gap is not None, extrapolation_ratio=ratio,
                history_steps=[h["step"] for h in self.history], selector_calls=0,
                execution_audit=c["execution_audit"]),
            model_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
        self.pending = result
        self.round += 1
        return result

    def acknowledge(self, executed, quaternions=None):
        if self.pending is None:
            raise ValueError("No proposal")
        executed = np.asarray(executed, np.float32)
        if not 0 < len(executed) <= self.config["R"] or not np.array_equal(executed, self.pending["actions"][:len(executed)]):
            raise ValueError("Execution mismatch")
        self.step += len(executed)
        self.pending = None
        return {}
