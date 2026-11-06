"""Bounded full-KV-ready tail probe. No publication/lease/runtime changes.

Only one instrumented Prefill graph and at most one eager validation KV exist.
Every replay and every KV read waits for full completion, never KV-ready.
"""
import csv
import ctypes
import gc
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma

from preparations.idea2.experiments import load_backend, native_raw
from preparations.idea2.graph_sampler import digest_cache
from preparations.idea2.openpi_backend import Prepared, cache_tensors
from preparations.idea2.runtime_graph import DynamicCore, input_tensors, stage_tensor

ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).read_bytes()
CHECKPOINT = 'shared/checkpoints/openpi/pi05_libero_pytorch_idea2'
GPU_UUID = os.environ.get('CEWP_DIAG_GPU_UUID', 'GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1')


def active_gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader'],text=True)
    return [line for line in output.splitlines() if line.startswith(GPU_UUID+',')]


def stats(values):
    return dict(mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(min(values)), maximum=float(max(values)))


class ExternalEvents:
    def __init__(self):
        runtime = Path(torch.__file__).parent.parent/'nvidia/cuda_runtime/lib/libcudart.so.12'
        self.library_path = str(runtime.resolve())
        self.lib = ctypes.CDLL(str(runtime))
        self.record = self.lib.cudaEventRecordWithFlags
        self.record.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
        self.record.restype = ctypes.c_int

    def mark(self, event, stream):
        code = self.record(event.cuda_event, stream.cuda_stream, 0x01)
        if code:
            raise RuntimeError(f'cudaEventRecordWithFlags external failed: {code}')


def make_events(stream, count):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for event in events:
        event.record(stream)  # Initialize native handles before capture.
    stream.synchronize()
    return events


def event_semantics(external, stream):
    start, middle, end, outer_start, outer_end = make_events(stream, 5)
    with torch.cuda.stream(stream):
        value = torch.zeros(1, device='cuda')
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        external.mark(start, stream)
        value.add_(1)
        torch.cuda._sleep(2500000)
        external.mark(middle, stream)
        value.add_(1)
        torch.cuda._sleep(2500000)
        external.mark(end, stream)
    rows = []
    for i in range(3):
        with torch.cuda.stream(stream):
            outer_start.record(stream)
            graph.replay()
            outer_end.record(stream)
        outer_end.synchronize()
        row = dict(iteration=i, before_to_start_ms=outer_start.elapsed_time(start),
                   first_ms=start.elapsed_time(middle), second_ms=middle.elapsed_time(end),
                   end_to_after_ms=end.elapsed_time(outer_end), value=float(value.item()))
        assert row['before_to_start_ms'] >= 0 and row['end_to_after_ms'] >= 0
        assert row['first_ms'] > 0 and row['second_ms'] > 0 and row['value'] == 2*(i+1)
        rows.append(row)
    return dict(passed=True, external_flag=1, replay_rows=rows,
                meaning='Captured external event records update their timestamp on every replay, bracketed by fresh outside events. Artificial sleep exists only in this API self-test.')


@torch.no_grad()
def main():
    assert not active_gpu_processes(), 'Selected GPU must be idle before this diagnostic starts'
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == GPU_UUID.removeprefix('GPU-')
    stream = torch.cuda.Stream()
    external = ExternalEvents()
    result = dict(status='running', torch=torch.__version__, cuda=torch.version.cuda,
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  gpu=torch.cuda.get_device_name(0), checkpoint=CHECKPOINT,
                  cuda_runtime=external.library_path,
                  torch_event_has_external='external' in inspect.signature(torch.cuda.Event).parameters,
                  semantics=event_semantics(external, stream), source_sha256=hashlib.sha256(SOURCE).hexdigest(),
                  scope='Single isolated full original-computation Prefill graph. Dynamic inputs, all full-end waits. Tail is event elapsed headroom, not GPU busy, overlap, request speedup or a changed publication protocol.')
    (ROOT/'loaded_source.py').write_bytes(SOURCE)
    snapshots = ROOT/'source'
    snapshots.mkdir(exist_ok=True)
    paths = [Path(inspect.getfile(DynamicCache)), Path(inspect.getfile(modeling_gemma)),
             Path('preparations/idea2/runtime_graph.py'), Path('preparations/idea2/graph_sampler.py'),
             Path('preparations/idea2/openpi_backend.py'), Path(inspect.getfile(torch.cuda.Event))]
    result['dependency_source_sha256'] = {}
    for path in paths:
        data=path.read_bytes()
        (snapshots/path.name).write_bytes(data)
        result['dependency_source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()
    (ROOT/'result.json').write_text(json.dumps(result,indent=2))
    print('External event self-test passed; loading pi05 on '+GPU_UUID, flush=True)
    backend = load_backend(CHECKPOINT, model_config='pi05_libero', num_steps=1)
    assert len(active_gpu_processes())==1, 'Another process appeared on selected GPU during model loading'
    bank_path=Path('results/idea1_v3_1/input_bank.pkl')
    bank=pickle.loads(bank_path.read_bytes())
    records=[r for r in bank if r['instruction']==bank[0]['instruction']]
    result['bank_sha256']=hashlib.sha256(bank_path.read_bytes()).hexdigest()
    prepared=[backend.prepare(native_raw(r)) for r in records]
    torch.cuda.synchronize()  # Cold input preparation, excluded from all timings.
    first=prepared[0]
    with torch.cuda.stream(stream):
        static=Prepared(None,[x.clone() for x in first.images],[x.clone() for x in first.image_masks],
                        first.tokens.clone(),first.token_masks.clone(),first.state.clone())
        core=DynamicCore(backend,static)
        for _ in range(3):
            temporary=core.build_prefix(static)
            stream.synchronize()
            del temporary
    gc.collect()
    start, ready, end, outside_end = make_events(stream,4)
    layers=backend.model.paligemma_with_expert.paligemma.language_model.layers
    last_layer=len(layers)-1
    original_update=DynamicCache.update
    calls=[]
    update_pointers=[]

    def audited_update(cache,key,value,layer_idx,cache_kwargs=None):
        output=original_update(cache,key,value,layer_idx,cache_kwargs)
        calls.append(layer_idx)
        if layer_idx==last_layer:
            external.mark(ready,stream)
            update_pointers.extend([x.data_ptr() for x in output])
        return output

    graph=torch.cuda.CUDAGraph()
    try:
        DynamicCache.update=audited_update
        with torch.cuda.graph(graph,stream=stream):
            external.mark(start,stream)
            prefix=core.build_prefix(static)
            external.mark(end,stream)
    finally:
        DynamicCache.update=original_update
    assert calls==list(range(len(layers)))
    assert update_pointers==[x.data_ptr() for x in prefix.cache[last_layer]]
    addresses=[x.data_ptr() for x in cache_tensors(prefix.cache)]
    result['capture']=dict(last_layer=last_layer,update_call_order=calls,
                           last_update_output_addresses_equal_final_cache=True,
                           update_restored=DynamicCache.update is original_update,
                           graph_pool=str(graph.pool()),captures=1)

    def replay(source):
        with torch.cuda.stream(stream):
            for target,value in zip(input_tensors(static),input_tensors(source),strict=True):
                stage_tensor(target,value)
            host_start=time.perf_counter()
            graph.replay()
            outside_end.record(stream)
            submit_ms=(time.perf_counter()-host_start)*1000
        outside_end.synchronize()  # Always full completion; never wait on ready.
        return dict(full_event_ms=start.elapsed_time(end),ready_event_ms=start.elapsed_time(ready),
                    kv_ready_to_end_ms=ready.elapsed_time(end),
                    end_to_external_done_ms=end.elapsed_time(outside_end),cpu_graph_submit_ms=submit_ms)

    for i in range(10):
        replay(prepared[i%len(prepared)])
    result['gpu_processes_before_timing']=active_gpu_processes()
    assert len(result['gpu_processes_before_timing'])==1
    rows=[]
    for i in range(100):
        row=dict(iteration=i,input_index=i%len(prepared),**replay(prepared[i%len(prepared)]))
        assert row['kv_ready_to_end_ms'] > 0
        assert abs(row['full_event_ms']-row['ready_event_ms']-row['kv_ready_to_end_ms']) < .01
        rows.append(row)
    result['gpu_processes_after_timing']=active_gpu_processes()
    assert result['gpu_processes_before_timing'][0].split(',')[1]==result['gpu_processes_after_timing'][0].split(',')[1]
    assert len(result['gpu_processes_after_timing'])==1
    with (ROOT/'measurements.csv').open('w') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result['summary']={k:stats([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms')}
    result['tail_fraction_mean']=float(np.mean([r['kv_ready_to_end_ms']/r['full_event_ms'] for r in rows]))
    print('100 dynamic-input replays completed; validating full KV against original eager',flush=True)
    validations=[]
    for i,source in enumerate(prepared):
        replay(source)
        actual=digest_cache(prefix)
        reference=backend.build_prefix(source)
        expected=digest_cache(reference)
        validations.append(dict(input_index=i,kv_digest_equal=actual==expected,digest=actual))
        del reference
        assert validations[-1]['kv_digest_equal']
    assert addresses==[x.data_ptr() for x in cache_tensors(prefix.cache)]
    result.update(status='complete',measurements=100,warmup=10,dynamic_inputs=len(prepared),
                  validation=validations,distinct_kv_digests=len({r['digest'] for r in validations}),
                  static_addresses_unchanged=True,steady_recaptures=0,max_complete_kv_snapshots=2,
                  waiting_protocol='Every iteration synchronizes outside_end after the complete graph. No reader waits on ready; no Pool, Lease, epoch, generation or publication implementation is modified.')
    (ROOT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
