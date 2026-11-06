"""Continuous-observation admission; the frozen two-slot GPU runtime is reused.

Environment control owns action selection. This single Host owns every Pool
mutation. Prediction ids and actuator chunk ids are deliberately distinct.
"""
import time

import numpy as np
import torch

from preparations.idea1_v2_1.runtime_client import input_digest, model_seed
from preparations.idea2.experiments import native_raw
from preparations.idea2.progressive_kv_policy import LEADS
from preparations.idea2.runtime_graph import layout
from results.idea2.online_encoder_20261025.policy import OnlinePolicy

ARMS = ("fresh18_no_encoder", "sm16_no_encoder", "sm16_encoder")


class ContinuousPolicy(OnlinePolicy):
    def __init__(self, backend, arm):
        self.last_input_step = None
        self.previous_capture = None
        super().__init__(backend, arm)

    @property
    def protocol(self):
        return dict(super().protocol, execution="continuous_progressive_cuda_graph",
                    real_time=True, control_hz=20, camera_hz=20,
                    observation_lag_control_steps=None, nominal_observation_lag_ms=None,
                    clock="MuJoCo owner paces native20Hz steps; inference continuously accepts actual later observations",
                    action_selection="latest completed prediction for next actuator chunk, only at R5 boundaries",
                    noise_identity="original model_seed(suite,task,init,target_chunk_id); all consumed chunk i uses seed(i)",
                    queue_capacity=1, tail_actions_used=False, lookahead_wait_limit_ms=75,
                    ahead_input_preparation=True,
                    missed_deadline="wait for a valid chunk; no extra action or H10 tail; control gap recorded")

    def reset(self):
        self.last_input_step = None
        self.previous_capture = None
        return super().reset()

    @staticmethod
    def _seed(seed, suite, task, init, target):
        if type(target) is not int or target < 0 or seed != model_seed(suite, task, init, target):
            raise ValueError("Actuator chunk noise identity changed")

    @torch.no_grad()
    def predict(self, obs, instruction, suite, seed, task, init, request_id=0,
                target_chunk_id=0, input_step=0, input_capture_host=None, **kwargs):
        self._seed(seed, suite, task, init, target_chunk_id)
        if (request_id, target_chunk_id, input_step) != (0, 0, 0):
            raise ValueError("Bootstrap must be actuator chunk0 from initial observation")
        result = super().predict(obs, instruction, suite, seed, **kwargs)
        self.last_input_step = 0
        self.previous_capture = dict(step=0, host=input_capture_host)
        result.update(request_id=0, target_chunk_id=0, input_step=0, input_capture_host=input_capture_host,
                      observation_lag_control_steps=None)
        return result

    @torch.no_grad()
    def predict_ahead(self, obs, instruction, suite, seed, request_id, target_chunk_id,
                      input_step, task, init, exchange, input_capture_host=None, **unused):
        if self.needs_bootstrap or self.engine is None:
            raise ValueError("Fresh bootstrap required")
        if (suite, instruction) != self.task or request_id != self.request_index:
            raise ValueError("Prediction request crossed task, episode, or order")
        self._seed(seed, suite, task, init, target_chunk_id)
        if target_chunk_id == 0:
            raise ValueError("Online inference must target a later actuator chunk")
        begin = time.perf_counter()
        try:
            digest = input_digest(obs, instruction)
            staged = self.next_input
            if request_id == 1:
                if input_step != 0 or digest != self.bootstrap_hash:
                    raise ValueError("First online request must consume bootstrap observation")
            elif input_step <= self.last_input_step:
                raise ValueError("Only actual later camera observations may be admitted")
            if staged is not None and (digest, input_step) != (staged["hash"], staged["step"]):
                raise ValueError("Accepted ahead observation/Prepared identity changed")
            if staged is not None and input_capture_host != staged["capture_host"]:
                raise ValueError("Accepted ahead frame capture timestamp changed")
            prepared = staged["prepared"] if staged is not None and staged["prepared"] is not None else self.backend.prepare(native_raw(dict(obs=obs, instruction=instruction)))
            noise = self.backend.noise(seed)
            if layout(prepared, noise) != self.signature:
                raise ValueError("Layout changed; episode reset required")
            self.next_input = None
        except BaseException:
            self.reset()
            raise
        exchange_record = {}
        old_capture = self.previous_capture

        def lookahead():
            probe = torch.cuda.Event();probe.record(self.engine.graphs.ae_stream)
            wait_begin = time.perf_counter()
            with torch.cuda.nvtx.range("Continuous/await_real_next_camera_frame"):
                value = exchange(request_id)
            received = time.perf_counter()
            terminal = bool(value.get("terminal", value.get("done", False)))
            exchange_record.update(terminal=terminal, has_next=bool(value.get("has_next")),
                reason=value.get("reason"), observation_wait_host_ms=(received-wait_begin)*1000,
                ae_done_when_next_observation_received=probe.query(),
                next_observation_received_host=received,
                query_scope="server receive/deserialization; probe follows AE graph and output clone/read events")
            if not value.get("has_next"):
                return None
            capture_step = value["capture_step"]
            capture_host = value.get("capture_host")
            if terminal or type(capture_step) is not int or capture_step <= input_step:
                raise ValueError("Ahead frame must be an actual later, nonterminal control observation")
            if capture_host is None or capture_host > received + .001:
                raise ValueError("Ahead frame capture time is missing or lies in the future")
            if value.get("suite", suite) != suite or value.get("instruction", instruction) != instruction:
                raise ValueError("Ahead frame belongs to a different task")
            digest_next = input_digest(value["obs"], instruction)
            if value.get("input_hash", digest_next) != digest_next:
                raise ValueError("Ahead frame input digest changed during transport")
            prepare_begin = time.perf_counter()
            # Hoist the identical CPU/upload preparation in every arm. Only
            # the next Vision graph replay differs between encoder conditions.
            nxt = self.backend.prepare(native_raw(dict(obs=value["obs"], instruction=instruction)))
            exchange_record.update(next_capture_step=capture_step, next_capture_host=capture_host,
                next_input_hash=digest_next, next_prepare_host_ms=(time.perf_counter()-prepare_begin)*1000,
                ae_done_after_next_preparation=probe.query())
            self.next_input = dict(hash=digest_next, step=capture_step, capture_host=capture_host, prepared=nxt)
            return nxt

        try:
            normalized, record, nxt = self.engine.request(prepared, noise, lead=LEADS[self.mode],
                collect_events=False, lookahead=lookahead, next_encoder=self.want_encoder)
            if bool(nxt is not None) != bool(exchange_record["has_next"]):
                raise RuntimeError("Ahead ticket was lost or invented")
            normalized = np.asarray(normalized)
            state = prepared.state[0].detach().cpu().numpy()
            actions = np.asarray(self.backend.policy._output_transform(dict(state=state, actions=normalized.copy()))["actions"])
            if normalized.shape != (10,32) or actions.shape != (10,7) or not np.isfinite(normalized).all() or not np.isfinite(actions).all():
                raise RuntimeError("Invalid continuous action output")
            ready = time.perf_counter()
            record.update(online=exchange_record, online_request_id=request_id,
                target_chunk_id=target_chunk_id, input_step=input_step, input_capture_host=input_capture_host,
                old_kv_capture=old_capture,
                capture_to_response_build_ms=(ready-input_capture_host)*1000 if input_capture_host is not None else None)
            public = {k:v for k,v in record.items() if not k.startswith("_")}
            result = dict(actions=actions, normalized_actions=normalized, execute_length=5,
                generated_length=10, submitted_length=5, seed=seed, input_hash=digest,
                request=request_id, request_id=request_id, target_chunk_id=target_chunk_id,
                input_step=input_step, input_capture_host=input_capture_host, mode=self.mode, arm=self.arm,
                lead=LEADS[self.mode], protocol=self.protocol, episode=self.episode, bootstrap=False,
                graph_setup_ms=0., graph_recaptures=0, graph_kv_buffers=2,
                frontiers=record["frontiers"], actual_next_encoder=record["next_encoder"],
                online=exchange_record, progressive=public,
                snapshot=dict(epoch=record["epoch"],slot=record["slot"],generation=record["generation"]),
                timing=dict(response_build_host_ms=(ready-begin)*1000,
                    capture_to_response_build_ms=record["capture_to_response_build_ms"],
                    scope="includes causal camera exchange; not earliest GPU-only action availability"))
            self.request_index += 1
            self.last_input_step = input_step
            self.previous_capture = dict(step=input_step,host=input_capture_host)
            return result
        except BaseException:
            self.reset()
            raise
