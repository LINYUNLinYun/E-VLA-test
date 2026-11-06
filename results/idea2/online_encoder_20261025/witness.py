"""Offline, prepared-input GPU mechanism witness; NOT online/E2E/quality evidence.

Run under nsys --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture
--capture-range=cudaProfilerApi --capture-range-end=stop. No CUDA work is done
by --help or --cpu-check. The normal entrypoint requires an idle physical GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BANK = ROOT/'results/idea2/progressive_kv_20261025/same_task_bank.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    from preparations.idea2.progressive_kv_bench import source_hashes
    result = source_hashes()
    for relative in (
        'results/idea2/online_encoder_20261025/witness.py',
        'results/idea2/fresh_tail_20261025/adapter.py',
        'results/idea2/progressive_kv_20261025/resource_adapter.py',
        'preparations/idea2/green_context.py', 'preparations/idea2/external_events.py',
        'preparations/idea2/experiments.py',
        'policies/openpi/src/openpi/training/config.py',
        'policies/openpi/src/openpi/models_pytorch/pi0_pytorch.py',
        'policies/openpi/src/openpi/models_pytorch/gemma_pytorch.py',
    ):
        result[relative] = sha(ROOT/relative)
    return result


def inspect_bank(path):
    from preparations.idea1_v2_1.runtime_client import input_digest
    records = pickle.loads(Path(path).read_bytes())
    if len(records) < 9 or len({(r['suite'], r['instruction']) for r in records}) != 1:
        raise ValueError('Need >=9 recorded observations with one task/instruction')
    evidence = []
    for index, row in enumerate(records):
        actual = input_digest(row['obs'], row['instruction'])
        if row.get('input_hash') != actual:
            raise ValueError(f'Bank input hash mismatch at {index}')
        evidence.append(dict(bank_index=index, input_sha256=actual, source=row.get('source'),
                             recorded_seed=row.get('seed'), witness_noise_seed=1701+index))
    if len({r['input_sha256'] for r in evidence}) < 2:
        raise ValueError('Witness bank must contain changing observations')
    return records, evidence


def execute(args, result, save):
    """GPU locals must leave this frame before Green Context destruction."""
    import torch
    from preparations.idea2.experiments import load_backend, native_raw
    from preparations.idea2.layer_bench import gpu_snapshot, require_same_processes, within_tolerance
    from preparations.idea2 import progressive_kv as runtime
    from results.idea2.fresh_tail_20261025 import adapter

    records, inputs = inspect_bank(args.bank)
    result['inputs'] = inputs
    layouts, _ = adapter.install(16, 0, 1)
    result['resource_adapter'] = dict(ae_sm_min=16, encoder_after_step=1, actual_resources=layouts,
        sha256=sha(ROOT/'results/idea2/progressive_kv_20261025/resource_adapter.py'))
    result['fresh_tail_adapter'] = dict(ae_sm_min=16, fresh_tail_steps=0,
        encoder_after_step=1, actual_resources=layouts,
        sha256=sha(ROOT/'results/idea2/fresh_tail_20261025/adapter.py'))
    engine = None
    profiler_started = False
    try:
        backend = load_backend(args.checkpoint, model_config='pi05_libero', num_steps=10)
        prepared = [backend.prepare(native_raw(row)) for row in records]
        noises = [backend.noise(1701+i) for i in range(len(records))]
        # Input transfers/preprocessing complete before captures and profiling.
        torch.cuda.current_stream(backend.device).synchronize()
        engine = runtime.ProgressiveRuntime(backend, prepared[0], noises[0], profile=True)
        if len(engine.graphs.kv_addresses) != 2:
            raise RuntimeError('Witness requires exactly two complete KV slots')
        own = {r['pid'] for r in gpu_snapshot(args.gpu_uuid)['processes']}
        if len(own) != 1:
            raise RuntimeError('Expected exactly one owned Host PID after exclusive model load')
        require_same_processes(args.gpu_uuid, own)
        result.update(identity=backend.identity, graph_setup_ms=engine.graphs.setup_ms,
            kv_addresses=engine.graphs.kv_addresses,
            scratch_bytes=sum(x.numel()*x.element_size() for x in engine.graphs.shared_scratch),
            own_gpu_processes=sorted(own), prepared_input_count=len(prepared))

        def sequence(length, phase, verify=False):
            engine.reset(prepared[0])
            rows = []
            current = prepared[1]
            for index in range(length):
                source = index+1
                bank_index = source % len(prepared)
                want_next = index+1 < length
                future = prepared[(source+1) % len(prepared)] if want_next else None
                if current is not prepared[bank_index]:
                    raise RuntimeError('Same Prepared ticket identity was not retained')
                # This callback does no tensor allocation, transform or transfer.
                def lookahead():
                    return future
                output, row, nxt = engine.request(current, noises[bank_index], lead=3,
                    lookahead=lookahead if want_next else None, next_encoder=True,
                    verify=verify, collect_events=False)
                if nxt is not future:
                    raise RuntimeError('Lookahead returned a different Prepared object')
                if row['full_kv_slots'] != 2 or row['graph_recaptures'] != 0:
                    raise RuntimeError('Two-slot or steady capture invariant changed')
                row.update(phase=phase, pair=-1 if verify else 0, index=index, variant='encoder',
                    source_id=source, bank_index=bank_index, input_sha256=inputs[bank_index]['input_sha256'],
                    noise_seed=1701+bank_index, input_prepare_ms=0.0,
                    same_prepared_ticket_identity=True,
                    lookahead_origin='pre-existing adjacent recorded bank observation, fully prepared before profiling')
                if verify and not within_tolerance(row['numerical']['matched_schedule'], 1e-5, 1e-6):
                    raise AssertionError(f'Matched-schedule oracle failed: {row["numerical"]}')
                rows.append(row)
                current = nxt
                del output
            engine.drain()
            for row in rows:
                engine.finish_record(row)
            if phase != 'warmup':
                result['rows'].extend(rows)
            result.setdefault('groups', []).append(dict(phase=phase, count=length,
                tickets=engine.pool.ticket_diagnostics, next_encoder_requests=sum(r['next_encoder'] for r in rows)))
            engine.records.clear()

        result['status'] = 'warming'; save()
        sequence(20, 'warmup')
        sequence(2, 'verification', verify=True)
        result['oracle'] = dict(passed=True, comparisons=2, reference='serial observed same per-round old/new KV schedule',
            max_abs=max(r['numerical']['matched_schedule']['max_abs'] for r in result['rows']),
            max_mae=max(r['numerical']['matched_schedule']['mae'] for r in result['rows']))
        require_same_processes(args.gpu_uuid, own)
        torch.cuda.reset_peak_memory_stats()
        result['status'] = 'profiling'; save()
        torch.cuda.cudart().cudaProfilerStart(); profiler_started = True
        try:
            sequence(8, 'profile')
        finally:
            torch.cuda.cudart().cudaProfilerStop(); profiler_started = False
        result.update(gpu_after=require_same_processes(args.gpu_uuid, own),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), source_after=sources())
        result['source_unchanged'] = result['source_after'] == result['source_before']
        if not result['source_unchanged']:
            raise RuntimeError('Witness sources changed during execution')
        result['status'] = 'measured_pending_cleanup'
    finally:
        if profiler_started:
            torch.cuda.cudart().cudaProfilerStop()
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--checkpoint', default=str(ROOT/'shared/checkpoints/openpi/pi05_libero_pytorch_idea2'))
    parser.add_argument('--bank', default=str(DEFAULT_BANK))
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_check:
        records, evidence = inspect_bank(args.bank)
        print(json.dumps(dict(status='passed',cpu_only=True,gpu_launched=False,
            records=len(records),distinct_inputs=len({r['input_sha256'] for r in evidence}),
            source_hashes=len(sources()),bank_sha256=sha(args.bank))))
        return 0
    if not args.output or not args.gpu_uuid:
        parser.error('--output and --gpu-uuid are required for an explicitly scheduled GPU witness')
    from preparations.idea2.layer_bench import require_idle_gpu
    dest=Path(args.output).resolve(); dest.mkdir(parents=True,exist_ok=False)
    result=dict(status='starting',args={**vars(args),'profile':True,'profile_lead':3,'profile_variant':'encoder'},
        command=shlex.join([sys.executable,'-m','results.idea2.online_encoder_20261025.witness',*sys.argv[1:]]),
        source_before=sources(),bank_sha256=sha(args.bank),rows=[],performance_eligible=False,
        tolerance=dict(max_abs=1e-5,mae=1e-6),
        timing_scope='Offline prepared-input GPU mechanism witness only. Input CPU transforms/GPU transfers, warmup and capture excluded. Runtime CPU spans include output readback and profiling overhead; no decoded action or online arrival latency claim.',
        input_origin='Existing recorded same-task bank, not a newly captured online future observation. This is intentionally not causal closed-loop/E2E/quality evidence.',
        semantics='Full pi05 D10/H10, SM16/112, lead3 progressive mixed old/new KV, tail0, next Encoder after AE step1 and current VLM; two complete KV slots.',
        expected_profile_requests=8,expected_next_encoder_intersections=7)
    def save(): (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    failure=None
    try:
        result['gpu_before']=require_idle_gpu(args.gpu_uuid)
        execute(args,result,save)
    except BaseException:
        failure=traceback.format_exc();result.update(status='failed',error=failure)
    # No execute traceback/frame remains live here: release all caller tensors
    # before destroying Green Contexts, including in the failure path.
    if 'results.idea2.fresh_tail_20261025.adapter' in sys.modules:
        try:
            adapter=sys.modules['results.idea2.fresh_tail_20261025.adapter']
            result['cleanup']=adapter.finish_contexts()
        except BaseException:
            result.update(status='failed',cleanup_error=traceback.format_exc());failure=failure or result['cleanup_error']
    if not failure:
        result['status']='passed'
    save()
    print(json.dumps(dict(status=result['status'],output=str(dest),oracle=result.get('oracle'),cleanup=result.get('cleanup'))))
    return 1 if failure else 0


if __name__=='__main__':raise SystemExit(main())
