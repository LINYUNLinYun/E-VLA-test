"""Exercise real actor selection without CUDA or MuJoCo."""
import json
import time
import numpy as np

from results.idea2.continuous_encoder_20261026.client import Mailbox
from results.idea2.continuous_encoder_20261026.cpu_selftest import frame, candidate
from .client import execute_actor


def run_trial(mode, source=2, limit=10, second_source=None):
    initial = frame(0)
    box = Mailbox(initial)
    box.first_advance = time.perf_counter()
    actions, rows, frames = [], [], {0: initial}
    count = [0]
    def step(action):
        count[0] += 1
        actions.append(action[0])
        return {}, 0., False, {}
    def capture(index, obs, action, metadata):
        frames[index] = frame(index)
        box.publish_frame(frames[index])
        rows.append(metadata)
        if index == 5 or (index == 10 and second_source is not None):
            target = index//5
            chunk = candidate(target, target, frames[source if index == 5 else second_source])
            chunk.actions[:] = np.arange(10, dtype=np.float32)[:, None]
            box.offer(chunk)
    state = {}
    count, done, chunks = execute_actor(box, candidate(0, 0, initial), limit, step, capture,
                                       period=.00001, alignment=mode, state=state)
    return actions, rows, chunks, state


def run():
    checks = []
    actions, rows, chunks, state = run_trial("off")
    assert actions == [0]*5+list(range(5))
    assert rows[5]["alignment_residual_steps"] == 3
    checks.append("off reproduces first-five action selection with explicit residual delay")
    actions, rows, chunks, state = run_trial("control_steps")
    assert actions == [0]*5+list(range(3, 8))
    assert all(r["alignment_residual_steps"] == 0 for r in rows)
    assert [c.record["executed_length"] for c in chunks] == [5, 5]
    assert chunks[1].record["skipped_action_prefix"] == 3
    checks.append("three-step age executes indices 3 through 7; no padding; R5 retained")
    actions, rows, chunks, state = run_trial("control_steps", source=0)
    assert actions[5:] == list(range(5, 10))
    assert rows[-1]["action_index"] == 9
    checks.append("five-step offset uses exactly the remaining H10 horizon")
    actions, rows, chunks, state = run_trial("control_steps", source=2, limit=15, second_source=3)
    assert len(actions) == 10 and state["horizon_exhausted"]
    assert state["requested_offset"] == 7 and len(chunks) == 2
    checks.append("insufficient horizon stops without executing or extending an action block")
    actions, rows, chunks, state = run_trial("control_steps", source=2, limit=8)
    assert actions[5:] == [3, 4, 5] and len(actions) == 8
    checks.append("native action limit truncates R5 without reading beyond H10")
    assert "torch" not in __import__("sys").modules and "libero" not in __import__("sys").modules
    return dict(status="passed", checks=checks, gpu_or_environment_started=False)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
