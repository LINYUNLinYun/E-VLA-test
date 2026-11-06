"""Future resident wrapper around the pinned experiment10 Backend.

No server or experiment imports this module. Numerical and memory gates on
actual GPUs are still required. Imports and CPU fixtures do not load torch.
"""
import hashlib
from pathlib import Path

from .memory_floor import MemoryRetainer, invariant_storage_bytes, plan_retention
from .prompt_runtime import PromptRuntimeOwner


ROOT = Path(__file__).resolve().parents[3]
BACKEND_PATH = ROOT / 'policies/realtime-vla-flash/scripts/experiment10/backend.py'
BACKEND_SHA256 = '569a321519100339a588b6b22916bf1ec10e04ee9b3ebf8fca477678ed656a1f'


def retention_factory(*, measured_working_bytes, available_bytes):
    """Use separately measured profile/capacity; never infer them from a mask."""
    def build(owner):
        tensors = owner.invariant_tensors()
        plan = plan_retention(measured_working_bytes=measured_working_bytes,
            invariant_storage_bytes=invariant_storage_bytes(tensors), available_bytes=available_bytes)
        return MemoryRetainer(tensors, plan)
    return build


class ResidentBackendAdapter:
    def __init__(self, wrapped_backend, *, build_retainer,
                 build_owner=PromptRuntimeOwner, captured_graph_caches=()):
        if hashlib.sha256(BACKEND_PATH.read_bytes()).hexdigest() != BACKEND_SHA256:
            raise RuntimeError('Frozen backend changed; review a new resident adapter')
        if wrapped_backend.runtime is not None or wrapped_backend.length is not None:
            raise ValueError('Install the adapter before the first model preparation')
        if not callable(build_retainer):
            raise ValueError('An explicit reviewed retention factory is required')
        self._wrapped = wrapped_backend
        self._build_owner, self._build_retainer = build_owner, build_retainer
        self._other_graph_caches = tuple(captured_graph_caches)
        self._owner = self._retainer = None
        self._episode_active = False
        self._episode_prompt = None
        self._failed = False
        self.events = []
        self.model_initializations = 0

    def __getattr__(self, name):
        # Existing adapters read runtime, inputs, transforms, embeddings,
        # schedules and graph caches; their ownership remains in the wrapper.
        return getattr(self._wrapped, name)

    @staticmethod
    def _acknowledged(pending_actions):
        if type(pending_actions) is not int or pending_actions != 0:
            raise ValueError('Previous policy proposal must be fully acknowledged')

    def begin_episode(self, *, pending_actions):
        self._acknowledged(pending_actions)
        if self._failed:
            raise RuntimeError('Resident adapter failed; further inference is forbidden')
        if self._episode_active:
            raise RuntimeError('Finish the current episode before opening another')
        if self._owner is not None:
            self.assert_ready()
        self._episode_active = True
        self._episode_prompt = None

    def finish_episode(self, *, pending_actions):
        self._acknowledged(pending_actions)
        if not self._episode_active:
            raise RuntimeError('No active episode')
        self.assert_ready()
        self._episode_active = False
        self._episode_prompt = None
        # Model weights, scratch runtime, schedules and floor all remain live.

    def assert_ready(self):
        if self._failed or self._owner is None or self._retainer is None:
            raise RuntimeError('Resident adapter is not ready')
        try:
            if self._wrapped.runtime is not self._owner.runtime:
                raise RuntimeError('Wrapped backend replaced the resident runtime')
            if self._wrapped.length != self._owner.runtime.prompt_len:
                raise RuntimeError('Wrapped prompt length differs from resident state')
            self._owner.assert_ready()
            self._retainer.assert_held()
        except Exception:
            self._failed = True
            raise

    def prepare(self, observation, meta, noise_array):
        if self._failed or not self._episode_active:
            raise RuntimeError('Prepare requires a healthy active resident episode')
        key = (meta['suite'], meta['task_description'])
        if self._episode_prompt is not None and key != self._episode_prompt:
            raise ValueError('A prompt must not change within an episode')
        embedding = self._wrapped.embeddings[key]
        length = len(embedding)
        try:
            if self._owner is None:
                self._wrapped.prepare(observation, meta, noise_array)
                if self._wrapped.runtime is None or self._wrapped.length != length:
                    raise RuntimeError('Initial frozen preparation did not create the expected runtime')
                self._owner = self._build_owner(self._wrapped.runtime)
                self._retainer = self._build_retainer(self._owner)
                self._retainer.retain()
                self.model_initializations += 1
                self.events.append(dict(kind='model_and_floor_initialized', prompt_length=length))
            else:
                self.assert_ready()
                if length != self._wrapped.length:
                    if self._episode_prompt is not None:
                        raise RuntimeError('Scratch replacement is only allowed before the first episode request')
                    event = self._owner.rebind(embedding, episode_boundary=True,
                        graph_caches=(self._wrapped.graphs, *self._other_graph_caches))
                    self._wrapped.length = length
                    self._wrapped.parity_lengths.clear()
                    self._retainer.assert_held()
                    self.events.append(dict(kind='prompt_scratch_rebound', **event))
                # length is already correct, so the pinned method bypasses its
                # runtime destruction / empty_cache / Pi0Inference branch.
                # It still applies the exact original embedding/input transforms.
                self._wrapped.prepare(observation, meta, noise_array)
            self._episode_prompt = key
            self.assert_ready()
        except Exception:
            self._failed = True
            raise
