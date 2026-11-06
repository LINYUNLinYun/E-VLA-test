"""Declared pi0 execution patch: two complete original Backend runtime states."""
from collections import OrderedDict
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time
import types


STATE = ('runtime', 'length', 'graphs', 'schedules', 'parity_lengths', 'inputs', 'transformed')


def function_identity(function):
    function = getattr(function, '__func__', function)
    return dict(module=function.__module__, qualname=function.__qualname__,
        source_file=inspect.getsourcefile(function),
        source_sha256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest())


class PromptLengthCache:
    def __init__(self, backend, adapter_schedules, capacity=2, original_prepare=None, clock=time.time):
        if capacity != 2:
            raise ValueError('This execution revision permits exactly two lengths')
        if backend.runtime is not None:
            raise ValueError('Install the cache before the first runtime/graph allocation')
        self.backend, self.adapter_schedules = backend, adapter_schedules
        self.original_prepare = original_prepare or type(backend).prepare
        self.capacity, self.slots, self.clock = capacity, OrderedDict(), clock
        self.last_event = None
        self.calls = 0

    def _clear_visible(self):
        self.backend.runtime = None
        self.backend.length = None
        self.backend.graphs = OrderedDict()
        self.backend.schedules = {}
        self.backend.parity_lengths = set()
        self.backend.inputs = None
        self.backend.transformed = None

    def _evict(self):
        length, slot = self.slots.popitem(last=False)
        runtime_id = id(slot.runtime)
        for key in list(self.adapter_schedules):
            if key[0] == runtime_id:
                del self.adapter_schedules[key]
        # Drop all graph/weight/buffer and dispatcher references before allocating
        # the replacement, never briefly retaining three complete runtimes.
        self._clear_visible()
        slot.graphs.clear()
        slot.schedules.clear()
        slot.parity_lengths.clear()
        slot.runtime = None
        slot.inputs = None
        slot.transformed = None
        return dict(prompt_length=length, runtime_id=runtime_id)

    def prepare(self, observation, meta, noise_array):
        key = len(self.backend.embeddings[meta['suite'], meta['task_description']])
        self.calls += 1
        hit, evicted = key in self.slots, []
        if hit:
            slot = self.slots.pop(key)
            self.slots[key] = slot
        else:
            if len(self.slots) == self.capacity:
                evicted.append(self._evict())
            slot = copy.copy(self.backend)
            slot.__dict__.pop('prepare', None)
            slot.runtime, slot.length = None, None
            slot.graphs, slot.schedules, slot.parity_lengths = OrderedDict(), {}, set()
            slot.inputs, slot.transformed = None, None
            self.slots[key] = slot
        event = dict(dispatch=self.calls, timestamp=self.clock(), prompt_length=key,
            hit=hit, miss=not hit, evictions=evicted, resident_lengths=list(self.slots), capacity=2)
        try:
            # Original prepare always refreshes language embedding, observation
            # transforms and staged image/state/noise. Prefix features are not cached.
            result = self.original_prepare(slot, observation, meta, noise_array)
            for name in STATE:
                setattr(self.backend, name, getattr(slot, name))
            event['runtime_id'] = id(slot.runtime)
            return result
        except BaseException as error:
            event['error'] = repr(error)
            raise
        finally:
            self.last_event = event


def install(policy, output_dir, revision_hash):
    from common import atomic_json, sha
    if policy.model_id != 'pi0':
        raise ValueError('Prompt cache patch is pi0-only')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = policy.pi0.backend
    original_predict = policy.predict
    prepare_identity = function_identity(type(backend).prepare)
    predict_identity = function_identity(original_predict)
    controller = PromptLengthCache(backend, policy.pi0.schedules)
    backend.prepare = types.MethodType(lambda owner, *args, **kwargs: controller.prepare(*args, **kwargs), backend)
    atomic_json(output_dir/'runtime_execution_patch.json', dict(schema=1,
        kind='bounded_prompt_length_runtime_cache', revision_manifest_sha256=revision_hash,
        capacity=2, original_prepare=prepare_identity, original_predict=predict_identity,
        patch_source=str(Path(__file__).resolve()), patch_source_sha256=sha(__file__),
        policy='original Backend per-length complete state; no observation/KV/mask feature reuse; eager original kernels'))
    def predict(owner, *args, **kwargs):
        row = original_predict(*args, **kwargs)
        event = dict(controller.last_event)
        startup = row['audit'].get('startup_events', [])
        event.update(call=row['call'], input_hash=row['input_hash'], noise_hash=row['noise_hash'],
            seed=row['seed'], skip=row['audit']['skip'], schedule=row['audit']['schedule'],
            startup_forward_calls=sum(item['forward_calls'] for item in startup),
            startup_action_NFE=sum(item['action_NFE'] for item in startup),
            prediction_path=row['prediction_path'],
            cuda_peak_allocated_mib=policy.torch.cuda.max_memory_allocated()/1024**2,
            cuda_peak_reserved_mib=policy.torch.cuda.max_memory_reserved()/1024**2)
        if event['cuda_peak_reserved_mib'] >= 23*1024:
            raise RuntimeError('CUDA allocator peak reached the reviewed cache memory cap')
        if event['hit'] and event['startup_forward_calls'] != 0:
            raise RuntimeError('A cache hit unexpectedly rebuilt a runtime')
        with (output_dir/'prompt_cache_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True)+'\n')
            stream.flush()
        row['execution_prompt_cache'] = event
        return row
    policy.predict = types.MethodType(predict, policy)
    return controller
