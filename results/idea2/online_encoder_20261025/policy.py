"""Causal next-Encoder experiment; one scheduling Host owns all Pool mutations.

The environment executes an already returned R5 chunk during the next request.
Consequently actions have an explicit one-chunk observation lag. This is a new
online protocol, not an unchanged synchronous LIBERO inference optimization.
"""
import time

import numpy as np
import torch

from preparations.idea1_v2_1.runtime_client import input_digest
from preparations.idea2.experiments import native_raw
from preparations.idea2.progressive_kv_policy import LEADS, ProgressivePolicy
from preparations.idea2.runtime_graph import layout
from results.idea2.fresh_tail_20261025 import adapter

CONDITIONS = {
    "fresh18_no_encoder": ("fresh18", 0, False),
    "fresh18_encoder": ("fresh18", 0, True),
    "sm16_no_encoder": ("lead3", 16, False),
    "sm16_encoder": ("lead3", 16, True),
}


class OnlinePolicy(ProgressivePolicy):
    def __init__(self, backend, arm):
        self.arm = arm
        self.next_input = None
        self.bootstrap_hash = None
        self.observation_exchange = None
        self.want_encoder = CONDITIONS[arm][2]
        self.resources, _ = adapter.install(CONDITIONS[arm][1], 0, 1)
        super().__init__(backend, CONDITIONS[arm][0])

    @property
    def protocol(self):
        return dict(super().protocol, arm=self.arm, next_encoder=self.want_encoder,
                    lookahead=True, real_time=False,
                    clock="causal LIBERO stepping overlaps inference; simulation is not wall-clock paced",
                    observation_lag_control_steps=5, nominal_observation_lag_ms=250,
                    execution="online_progressive_cuda_graph",
                    old_action_execution="only the first R5 of the previously returned H10 chunk",
                    observation_exchange="same environment, after the previous chunk's actual actions",
                    resource_adapter=dict(ae_sm_min=CONDITIONS[self.arm][1], actual_resources=self.resources))

    def configure(self, arm, **unused):
        if arm not in CONDITIONS:
            raise ValueError("Unknown online arm")
        self.close()
        self.arm = arm
        self.mode, sms, self.want_encoder = CONDITIONS[arm]
        self.resources, _ = adapter.install(sms, 0, 1)
        self.reset()
        return self.protocol

    def _diagnostics(self):
        if self.engine is None:
            return []
        self.engine.drain()
        result = []
        for record in self.engine.records:
            self.engine.finish_record(record)
            result.append({k: v for k, v in record.items() if not k.startswith("_")})
        self.engine.records.clear()
        return result

    def drain(self):
        return dict(records=self._diagnostics(), pending_ahead=self.next_input is not None,
                    episode=self.episode)

    def reset(self):
        self.next_input = None
        self.bootstrap_hash = None
        result = super().reset()
        return result

    def close(self):
        self.next_input = None
        return super().close()

    @torch.no_grad()
    def predict(self, obs, instruction, suite, seed, **kwargs):
        if not self.needs_bootstrap:
            raise ValueError("Use predict_ahead after the initial current-KV bootstrap")
        result = super().predict(obs, instruction, suite, seed, **kwargs)
        self.bootstrap_hash = result["input_hash"]
        result.update(arm=self.arm, observation_lag_control_steps=0, actual_next_encoder=False)
        return result

    @torch.no_grad()
    def predict_ahead(self, obs, instruction, suite, seed, request_id, input_step, exchange, **unused):
        if self.needs_bootstrap or self.engine is None:
            raise ValueError("A returned bootstrap action chunk is required before online execution")
        if (suite, instruction) != self.task or request_id != self.request_index:
            raise ValueError("Online request crossed task, episode, or request order")
        begin = time.perf_counter()
        try:
            digest = input_digest(obs, instruction)
            staged = self.next_input
            if staged is None and (request_id != 1 or input_step != 0 or digest != self.bootstrap_hash):
                raise ValueError("First online request must use the bootstrap observation; later requests need an accepted observation")
            if staged is not None:
                if (digest, input_step) != (staged["hash"], staged["step"]):
                    raise ValueError("Next request did not consume the actual accepted observation")
            prepared = staged["prepared"] if staged is not None and staged["prepared"] is not None else self.backend.prepare(native_raw(dict(obs=obs, instruction=instruction)))
            noise = self.backend.noise(seed)
            if layout(prepared, noise) != self.signature:
                raise ValueError("Online layout changed; reset before admitting another task")
            self.next_input = None
        except BaseException:
            self.reset()
            raise
        exchange_record = {}

        def lookahead():
            # Mark after the AE graph was submitted. Querying this event does
            # not wait, and exposes observations that arrived too late to hide AE.
            probe = torch.cuda.Event()
            probe.record(self.engine.graphs.ae_stream)
            wait_begin = time.perf_counter()
            with torch.cuda.nvtx.range("Online/await_real_next_observation"):
                value = exchange(request_id)
            received = time.perf_counter()
            exchange_record.update(observation_wait_host_ms=(received-wait_begin)*1000,
                                   ae_done_when_next_observation_received=probe.query(),
                                   next_observation_received_host=received,
                                   query_scope="server deserialized observation; AE graph, output clone and read events precede the probe")
            if not value.get("has_next"):
                exchange_record.update(terminal=True, next_capture_step=value["capture_step"])
                return None
            if value["capture_step"] != input_step + 5:
                # The first duplicate bootstrap input also advances five steps.
                raise ValueError("Expected one complete R5 advance for the next observation")
            if value.get("suite", suite) != suite or value.get("instruction", instruction) != instruction:
                raise ValueError("Next observation changed task")
            next_obs = value["obs"]
            next_digest = input_digest(next_obs, instruction)
            prepare_begin = time.perf_counter()
            nxt = self.backend.prepare(native_raw(dict(obs=next_obs, instruction=instruction))) if self.want_encoder else None
            exchange_record.update(terminal=False, next_capture_step=value["capture_step"],
                                   next_input_hash=next_digest,
                                   next_prepare_host_ms=(time.perf_counter()-prepare_begin)*1000,
                                   ae_done_after_next_preparation=probe.query())
            self.next_input = dict(hash=next_digest, step=value["capture_step"], prepared=nxt)
            return nxt

        try:
            normalized, record, nxt = self.engine.request(prepared, noise, lead=LEADS[self.mode],
                collect_events=False, lookahead=lookahead, next_encoder=self.want_encoder)
            if bool(nxt is not None) != bool(self.want_encoder and not exchange_record["terminal"]):
                raise RuntimeError("Ahead Prepared object was lost or invented")
            record.update(online=exchange_record, input_step=input_step, online_request_id=request_id)
            normalized = np.asarray(normalized)
            state = prepared.state[0].detach().cpu().numpy()
            actions = np.asarray(self.backend.policy._output_transform(dict(state=state, actions=normalized.copy()))["actions"])
            if normalized.shape != (10, 32) or actions.shape != (10, 7) or not np.isfinite(normalized).all() or not np.isfinite(actions).all():
                raise RuntimeError("Invalid online action output")
            public = {k: v for k, v in record.items() if not k.startswith("_")}
            result = dict(actions=actions, normalized_actions=normalized, execute_length=5,
                generated_length=10, submitted_length=5, seed=seed, input_hash=digest,
                mode=self.mode, arm=self.arm, lead=LEADS[self.mode], protocol=self.protocol,
                request=self.request_index, episode=self.episode, bootstrap=False,
                graph_setup_ms=0., graph_recaptures=0, frontiers=record["frontiers"],
                actual_next_encoder=record["next_encoder"], input_step=input_step,
                nominal_observation_lag_ms=250, observation_lag_control_steps=5,
                progressive=public, online=exchange_record,
                snapshot=dict(epoch=record["epoch"], slot=record["slot"], generation=record["generation"]),
                graph_kv_buffers=2,
                timing=dict(response_build_host_ms=(time.perf_counter()-begin)*1000,
                            scope="includes waiting for actual R5 advance/observation upload; not earliest GPU action readiness"))
            self.request_index += 1
            return result
        except BaseException:
            self.reset()
            raise
