"""Observation/version checks for future asynchronous execute-then-correct runs.

Executed robot actions are irreversible. Old Full18 outputs are rejected rather
than spliced onto a new physical state; this module does not run the async server.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FullRequest:
    episode: str
    request_id: int
    observation_id: int
    completed_actions: int
    observation_time_ms: float


class RequestLedger:
    def __init__(self, maximum_age_ms):
        if not math.isfinite(maximum_age_ms) or maximum_age_ms <= 0:
            raise ValueError('Positive frozen observation-age limit required')
        self.maximum_age_ms = maximum_age_ms
        self.reset(None)

    def reset(self, episode):
        self.episode = episode
        self.observation_id = -1
        self.completed_actions = 0
        self.observation_time_ms = None
        self.observation_completed_actions = None
        self.next_request = 0
        self.pending = {}

    def observe(self, observation_id, completed_actions, time_ms):
        if (self.episode is None or not isinstance(observation_id, int) or isinstance(observation_id, bool) or
            not isinstance(completed_actions, int) or isinstance(completed_actions, bool) or
            observation_id <= self.observation_id or completed_actions < self.completed_actions or
            not math.isfinite(time_ms) or time_ms < 0 or
            (self.observation_time_ms is not None and time_ms < self.observation_time_ms)):
            raise ValueError('Nonmonotonic/invalid observation or missing episode reset')
        self.observation_id = observation_id
        self.completed_actions = completed_actions
        self.observation_time_ms = time_ms
        self.observation_completed_actions = completed_actions

    def executed(self, actions):
        if not isinstance(actions, int) or isinstance(actions, bool) or actions < 1 or self.observation_id < 0:
            raise ValueError('Positive actual execution count required')
        self.completed_actions += actions

    def submit(self):
        if self.observation_id < 0:
            raise ValueError('No observation to submit')
        if self.completed_actions != self.observation_completed_actions:
            raise ValueError('Actions advanced after observation; obtain a fresh observation before replanning')
        request = FullRequest(self.episode, self.next_request, self.observation_id,
                              self.completed_actions, self.observation_time_ms)
        self.next_request += 1
        self.pending[request.request_id] = request
        return request

    def resolve(self, request, now_ms):
        if request.episode != self.episode:
            return False, 'wrong_episode_replan'
        registered = self.pending.get(request.request_id)
        if registered != request:
            return False, 'unknown_duplicate_or_mismatched_request'
        del self.pending[request.request_id]
        if not math.isfinite(now_ms) or now_ms < request.observation_time_ms:
            return False, 'invalid_completion_time'
        if request.completed_actions != self.completed_actions:
            return False, 'physical_state_advanced_replan_no_rollback'
        if request.observation_id != self.observation_id:
            return False, 'newer_observation_exists_replan'
        if now_ms-request.observation_time_ms > self.maximum_age_ms:
            return False, 'observation_expired_replan'
        return True, 'same_observation_same_execution_state'
