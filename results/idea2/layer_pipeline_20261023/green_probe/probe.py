"""Small one-process GPU capability probe. Never runs without --run-gpu.

Run only after the task owner coordinates an exclusive card. This is not a
model benchmark and makes no request-performance claim. Compiled .so files
contain CPU host wrappers plus one SM-ID diagnostic CUDA kernel.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from preparations.idea2.green_context import GreenContextPair, build_library
from preparations.idea2.external_events import ExternalEvents, initialized_events


ROOT=Path(__file__).resolve().parent


def compile_probe():
    output=ROOT/'build'/'smid_probe.so'
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-shared','-Xcompiler=-fPIC',
             '-arch=sm_89',str(ROOT/'smid_probe.cu'),'-o',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    (ROOT/'build'/'smid_build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return output,command


def processes(uuid):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader'],text=True)
    return [line for line in text.splitlines() if line.startswith(uuid+',')]


def difference(a,b):
    diff=(a.float()-b.float()).abs()
    return dict(max_abs=diff.max().item(),mae=diff.mean().item(),exact=torch.equal(a,b))


def sm_ids(tensor):
    assert tensor[-1].item()==0,'SM-ID diagnostic capacity exceeded'
    return torch.nonzero(tensor,as_tuple=True)[0].cpu().tolist()


@torch.no_grad()
def run_pair(library,kernel,min_sm):
    primary=torch.cuda.current_stream()
    base=torch.randn(128,256,device='cuda',dtype=torch.float32)*.02
    a=base.clone()
    b=torch.randn(256,256,device='cuda',dtype=torch.float32)*.02
    c=torch.randn(256,128,device='cuda',dtype=torch.float32)*.02
    inter=torch.empty(128,256,device='cuda')
    output=torch.empty(128,128,device='cuda')
    flags=[torch.zeros(4096,dtype=torch.int32,device='cuda') for _ in range(2)]
    input_ready=torch.cuda.Event()
    produced=torch.cuda.Event()
    done=torch.cuda.Event()
    torch.cuda.synchronize()
    pair=GreenContextPair(library,min_sm,device='cuda:0')
    result=dict(configuration=pair.info)
    graphs=[]

    def launch(index):
        code=kernel.idea2_launch_smid(flags[index].data_ptr(),flags[index].numel(),
                                     100000,4096,pair.streams[index].cuda_stream)
        if code:
            raise RuntimeError(f'cuda smid kernel launch returned {code}')

    try:
        sa,sb=pair.streams
        for _ in range(5):
            input_ready.record(primary)
            with torch.cuda.stream(sa):
                sa.wait_event(input_ready)
                torch.mm(a,b,out=inter)
                produced.record(sa)
            with torch.cuda.stream(sb):
                sb.wait_event(produced)
                torch.mm(inter,c,out=output)
                done.record(sb)
            done.synchronize()
        reference=(a@b)@c
        result['eager_fp32']=difference(reference,output)
        assert torch.allclose(reference,output,atol=1e-5,rtol=1e-4)

        # BF16 GEMM also consumes primary-context tensors through the green stream.
        a16,b16=a.bfloat16(),b.bfloat16()
        input_ready.record(primary)
        with torch.cuda.stream(sa):
            sa.wait_event(input_ready)
            actual16=a16@b16
            done.record(sa)
        done.synchronize()
        result['eager_bf16']=difference(a16@b16,actual16)
        assert torch.allclose(a16@b16,actual16,atol=1e-4,rtol=1e-2)

        # Validate the actual layer-pipeline mechanism: external record/wait
        # nodes in separate graphs, rather than only outside-graph Event waits.
        external=ExternalEvents()
        target=torch.empty_like(a)
        consumed=torch.empty_like(a)
        cross_ready,=initialized_events(sa,1)
        torch.cuda.synchronize()
        producer_graph,consumer_graph=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
        graphs.extend([producer_graph,consumer_graph])
        with torch.cuda.graph(producer_graph,stream=sa):
            torch.cuda._sleep(500000)
            target.copy_(a)
            external.mark(cross_ready,sa)
        with torch.cuda.graph(consumer_graph,stream=sb):
            external.wait(cross_ready,sb)
            consumed.copy_(target)
        for index in range(20):
            a.copy_(base*(1+index*.0625))
            input_ready.record(primary)
            with torch.cuda.stream(sa):
                sa.wait_event(input_ready)
                producer_graph.replay()
            with torch.cuda.stream(sb):
                consumer_graph.replay()
                done.record(sb)
            done.synchronize()
            assert torch.equal(a,consumed),f'External graph wait consumed stale input {index}'
        result['external_cross_graph_wait']=dict(passed=True,repetitions=20,
            delayed_producer_cycles=500000,changed_input_same_address=True,
            producer_launched_before_consumer=True,
            scope='Artificial sleep only in this capability diagnostic, not a model path.')
        producer_graph.reset()
        consumer_graph.reset()
        graphs.clear()

        # Independent SM-ID kernels on both partition streams, without a join.
        input_ready.record(primary)
        for i,stream in enumerate(pair.streams):
            with torch.cuda.stream(stream):
                stream.wait_event(input_ready)
                flags[i].zero_()
                launch(i)
        for stream in pair.streams:
            stream.synchronize()
        eager_ids=[sm_ids(value) for value in flags]
        result['eager_smid_sets']=eager_ids

        # Capture two private pools, then test ordinary cross-stream Event ordering
        # OUTSIDE capture. Green-context-wide RecordEvent/WaitEvent is not used.
        ga,gb=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
        graphs.extend([ga,gb])
        with torch.cuda.graph(ga,stream=sa):
            torch.mm(a,b,out=inter)
            flags[0].zero_()
            launch(0)
        sa.synchronize()
        with torch.cuda.graph(gb,stream=sb):
            torch.mm(inter,c,out=output)
            flags[1].zero_()
            launch(1)
        sb.synchronize()
        assert ga.pool()!=gb.pool()
        result['graph_pools']=[str(ga.pool()),str(gb.pool())]
        rows=[]
        for iteration in range(10):
            a.copy_(base*(1+iteration*.03125))
            input_ready.record(primary)
            with torch.cuda.stream(sa):
                sa.wait_event(input_ready)
                ga.replay()
                produced.record(sa)
            with torch.cuda.stream(sb):
                sb.wait_event(produced)
                gb.replay()
                done.record(sb)
            done.synchronize()
            expected=(a@b)@c
            row=dict(iteration=iteration,error=difference(expected,output),
                     smid_sets=[sm_ids(value) for value in flags])
            assert torch.allclose(expected,output,atol=1e-5,rtol=1e-4)
            rows.append(row)
        result['graph_replay_rows']=rows
        sets=[set(ids) for ids in eager_ids]
        result['observed_eager_sm_counts']=[len(ids) for ids in sets]
        result['eager_disjoint']=not bool(sets[0]&sets[1])
        result['observed_counts_match_resource']=result['observed_eager_sm_counts']==pair.info['sm_counts']
        result['graph_respects_observed_partition']=all(
            set(row['smid_sets'][i])<=sets[i] for row in rows for i in range(2))
        assert result['eager_disjoint']
        assert result['observed_counts_match_resource']
        assert result['graph_respects_observed_partition']
    finally:
        for graph in graphs:
            graph.reset()
        graphs.clear()
        gc.collect()
        pair.close()
    # Green contexts do not own or reset these primary-context allocations.
    result['primary_tensors_survive_close']=bool(torch.isfinite((a@b)@c).all().item())
    pair.close()  # Idempotent close.
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--first-sm-min',nargs='+',type=int,default=[32,64,96])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    library=build_library(ROOT/'build')
    probe_library,command=compile_probe()
    prepared=dict(library=str(library),probe_library=str(probe_library),
                  probe_build_command=command,cpu_only=True,cuda_initialized=torch.cuda.is_initialized())
    assert not prepared['cuda_initialized']
    (ROOT/'prepared_probe.json').write_text(json.dumps(prepared,indent=2))
    if args.prepare_only or not args.run_gpu:
        print(json.dumps(prepared,indent=2))
        return
    if not args.gpu_uuid or args.output is None:
        raise ValueError('--run-gpu requires --gpu-uuid and new --output directory')
    if processes(args.gpu_uuid):
        raise RuntimeError('Selected GPU has another process; do not run this probe')
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',torch=torch.__version__,runtime_cuda=torch.version.cuda,
                gpu_uuid=args.gpu_uuid,preparation=prepared,partitions=[],model_loaded=False,
                semantics='One process, primary-context tensors, two disjoint resources. Capability only; no model/latency claims.')
    path=args.output/'result.json'
    try:
        assert torch.cuda.device_count()==1,'Expose only the coordinated GPU'
        torch.cuda.set_device(0)
        actual=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        assert actual==args.gpu_uuid.removeprefix('GPU-')
        torch.backends.cuda.matmul.allow_tf32=False
        result['device']=torch.cuda.get_device_name(0)
        # Device properties alone may not instantiate a visible primary-context
        # compute process. Force one tiny allocation before the occupancy audit.
        context_marker=torch.zeros(1,device='cuda')
        torch.cuda.synchronize()
        result['processes_before']=processes(args.gpu_uuid)
        assert len(result['processes_before'])==1
        kernel=ctypes.CDLL(str(probe_library.resolve()))
        kernel.idea2_launch_smid.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_ulonglong,
                                         ctypes.c_uint,ctypes.c_void_p]
        kernel.idea2_launch_smid.restype=ctypes.c_int
        for minimum in args.first_sm_min:
            result['partitions'].append(run_pair(library,kernel,minimum))
            path.write_text(json.dumps(result,indent=2))
        result['processes_after']=processes(args.gpu_uuid)
        assert len(result['processes_after'])==1
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['status']='complete'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'smid_probe.cu',Path('preparations/idea2/green_context.py'))}
        path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
