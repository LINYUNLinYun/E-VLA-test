"""Failure-path checks: a rejected or failed input cannot strand an ahead ticket."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preparations.idea1_v2_1.runtime_client import input_digest
from .policy import OnlinePolicy


def main():
    obs = {"state": np.zeros(3, dtype=np.float32)}
    digest = input_digest(obs, "task")
    checks = []

    def policy(staged, index=2):
        p = OnlinePolicy.__new__(OnlinePolicy)
        p.needs_bootstrap = False;p.engine = object();p.task = ("suite", "task")
        p.request_index = index;p.next_input = staged;p.bootstrap_hash = digest
        reset_calls = []
        def reset():
            reset_calls.append(True);p.next_input = None
        def failed_noise(seed): raise RuntimeError("injected noise preparation failure")
        p.reset = reset;p.backend = SimpleNamespace(noise=failed_noise)
        return p, reset_calls

    p, reset = policy(dict(hash=digest, step=5, prepared=object()))
    try: p.predict_ahead(obs, "task", "suite", 7, 2, 5, lambda _: None)
    except RuntimeError as error: assert "injected" in str(error)
    else: raise AssertionError("Injected failure was swallowed")
    assert reset == [True] and p.next_input is None
    checks.append("noise failure resets accepted Prepared/ticket state")

    p, reset = policy(dict(hash="wrong", step=5, prepared=object()))
    try: p.predict_ahead(obs, "task", "suite", 7, 2, 5, lambda _: None)
    except ValueError as error: assert "accepted observation" in str(error)
    else: raise AssertionError("Mismatched staged identity was accepted")
    assert reset == [True] and p.next_input is None
    checks.append("mismatched next input rejected and reset")

    p, reset = policy(None, 1)
    try: p.predict_ahead(obs, "task", "suite", 7, 1, 5, lambda _: None)
    except ValueError as error: assert "bootstrap observation" in str(error)
    else: raise AssertionError("Non-bootstrap first online observation accepted")
    assert reset == [True]
    checks.append("first online observation must be bootstrap input at step0")
    value = dict(status="passed", gpu_operations=0, checks=checks)
    Path(__file__).with_suffix(".json").write_text(json.dumps(value, indent=2)+"\n")
    print(json.dumps(value))


if __name__ == "__main__": main()
