"""Client-only control-step alignment; the frozen GPU computation is reused.

At an R5 handoff in state k, a prediction from state s starts at index k-s.
This is prefix skipping, not RTC/inpainting or a guarantee of state agreement.
An insufficient H10 horizon terminates the scheduling trial without padding.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from results.idea2.continuous_encoder_20261026 import client as base

MODES = ("off", "control_steps")


def execute_actor(mailbox, bootstrap, limit, step_fn, publish_capture,
                  period=.05, chunks=None, *, alignment="control_steps", state=None):
    if alignment not in MODES:
        raise ValueError("Unknown alignment")
    if state is None:
        state = {}
    if chunks is None:
        chunks = []
    chunks.append(bootstrap)
    bootstrap.record.update(disposition="consumed", selected_host=time.perf_counter(),
                            boundary_wait_ms=0.)
    with mailbox.condition:
        deadline = time.perf_counter()+120
        while mailbox.first_advance is None:
            if mailbox.error:
                raise RuntimeError(mailbox.error)
            if time.perf_counter()>deadline:
                raise TimeoutError("First inference did not advance")
            mailbox.condition.wait(.1)
        next_deadline = mailbox.first_advance
    step = 0
    done = False
    chunk = bootstrap
    try:
        while step < limit and not done:
            age = step-chunk.frame.step
            if age < 0:
                raise ValueError("Cannot execute a prediction from a future state")
            offset = age if alignment == "control_steps" else 0
            count = min(5, limit-step)
            if offset+count > len(chunk.actions):
                chunk.record.update(disposition="horizon_exhausted", skipped_action_prefix=offset,
                                    execution_start_step=step, available_actions=max(0, len(chunk.actions)-offset))
                if chunks and chunks[-1] is chunk:
                    chunks.pop()
                state.update(horizon_exhausted=True, termination="prediction_horizon_exhausted",
                             stopped_at_step=step, input_step=chunk.frame.step, requested_offset=offset)
                break
            chunk.record.update(execution_start_step=step, observation_age_at_selection_steps=age,
                                skipped_action_prefix=offset, alignment=alignment)
            for position in range(count):
                with mailbox.condition:
                    if mailbox.error:
                        raise RuntimeError(mailbox.error)
                now = time.perf_counter()
                if next_deadline>now:
                    time.sleep(next_deadline-now)
                index = offset+position
                action = np.array(chunk.actions[index], dtype=np.float32, copy=True)
                application = time.perf_counter()
                if position == 0:
                    chunk.record["execution_start_host"] = application
                obs, _, done, _ = step_fn(action.tolist())
                captured = time.perf_counter()
                step += 1
                chunk.record["executed_length"] += 1
                publish_capture(step, obs, action, dict(
                    chunk_id=chunk.record["target_chunk_id"], consumed_request_id=chunk.record["request_id"],
                    seed=chunk.record["seed"], action_index=index, action_position_in_block=position,
                    skipped_action_prefix=offset, alignment=alignment,
                    input_step=chunk.frame.step, input_capture_host=chunk.frame.captured,
                    prediction_response_host=chunk.record["response_host"], application_host=application,
                    capture_host=captured, scheduled_application_host=next_deadline,
                    deadline_lateness_ms=max(0., application-next_deadline)*1000,
                    environment_step_ms=(captured-application)*1000,
                    observation_age_steps=step-1-chunk.frame.step,
                    alignment_residual_steps=step-1-chunk.frame.step-index,
                    observation_age_ms=(application-chunk.frame.captured)*1000,
                    response_to_application_ms=(application-chunk.record["response_host"])*1000,
                    action_sha256=base.original.value_digest(action)))
                next_deadline = max(next_deadline+period, application+period)
                if done:
                    break
            chunk.record.update(execution_end_step=step, execution_end_host=time.perf_counter())
            if done or step>=limit:
                break
            original_deadline = next_deadline
            remaining = original_deadline-time.perf_counter()
            if remaining>0:
                time.sleep(remaining)
            chunk = mailbox.take(chunk.record["target_chunk_id"]+1)
            selected = time.perf_counter()
            chunk.record["boundary_deadline_gap_ms"] = max(0., selected-original_deadline)*1000
            chunks.append(chunk)
            next_deadline = max(original_deadline, selected)
    finally:
        mailbox.stop()
    state.update(executed_steps=step, success=bool(done), alignment=alignment)
    return step, bool(done), chunks


def run_episode(args):
    state = dict(alignment=args.alignment, horizon_exhausted=False)
    client_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validate = base.validate_protocol
    write = base.write_json

    def validate_protocol(protocol, arm):
        validate(protocol, arm)
        if protocol.get("control_alignment") != args.alignment:
            raise ValueError("Server and client control alignment disagree")

    def write_json(path, value):
        if Path(path).name in ("identity.json", "environment_protocol.json", "complete.json", "result.json"):
            value = dict(value, control_alignment=args.alignment, alignment_client_sha256=client_sha)
            if Path(path).name in ("complete.json", "result.json") and state.get("horizon_exhausted"):
                value.update(termination="prediction_horizon_exhausted", horizon_exhausted=True)
        return write(path, value)

    try:
        with patch.object(base, "execute_actor", partial(execute_actor, alignment=args.alignment, state=state)), \
             patch.object(base, "validate_protocol", validate_protocol), patch.object(base, "write_json", write_json):
            result = base.run_episode(args)
        if state.get("horizon_exhausted"):
            result.update(termination="prediction_horizon_exhausted", horizon_exhausted=True)
        result.update(control_alignment=args.alignment, alignment_client_sha256=client_sha)
        return result
    finally:
        out = Path(args.output)
        if out.exists():
            write(out/"control_alignment.json", dict(state, client_sha256=client_sha,
                  method="offset=application_state_step-input_step; execute actions[offset:offset+R]",
                  horizon_policy="insufficient H10 terminates scheduling trial; no padding or R extension",
                  limitation="index alignment does not correct state error caused by previously executed actions"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--suite", choices=base.original.SUITES, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--init", type=int, required=True)
    p.add_argument("--arm", choices=("sm16_encoder", "fresh18_no_encoder"), required=True)
    p.add_argument("--alignment", choices=MODES, required=True)
    p.add_argument("--output", required=True)
    print(json.dumps(run_episode(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
