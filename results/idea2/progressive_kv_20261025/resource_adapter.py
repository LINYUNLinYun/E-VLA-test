"""Results-only optional resource/tail-timing ablation; frozen runtime untouched.

Use --ae-sm-min 16/32 for a single disjoint AE/Prefill Green Context split.
Use --encoder-after-step 7 to release next Vision later in AE's tail. This is
an Event dependency, never a Host sleep or dropped Prefill. Defaults preserve
the base runtime. Numerical oracle and all two-slot protocols remain in force.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time


def install(ae_sms=0, encoder_after_step=1):
    import torch
    from preparations.idea2 import progressive_kv as runtime
    Base = runtime.ProgressiveGraphs
    layouts=[]
    pending=[]

    class AdaptedGraphs(Base):
        def __init__(self, backend, prepared, noise, *, profile=False):
            self.resource_pair = None
            if not ae_sms:
                super().__init__(backend,prepared,noise,profile=profile)
                return
            from preparations.idea2.green_context import GreenContextPair, build_library
            self.backend,self.profile=backend,profile
            library=build_library(Path(__file__).resolve().parent/'green_build')
            self.resource_pair=GreenContextPair(library,ae_sms,device=backend.device)
            layouts.append(dict(self.resource_pair.info))
            self.ae_stream,self.prefix_stream=self.resource_pair.streams
            # Encoder and VLM never overlap each other in this experiment.
            self.vision_stream=self.prefix_stream
            self.external=runtime.ExternalEvents()
            self.slots=[]
            self.steps=backend.num_steps
            self.num_layers=len(backend.model.paligemma_with_expert.paligemma.language_model.layers)
            self.policy_constants={v:torch.tensor(v,device=backend.device,dtype=torch.int32) for v in (-1,0)}
            begin=time.perf_counter()
            initial=torch.cuda.Event();initial.record()
            for stream in self.streams:stream.wait_event(initial)
            try:
                for _ in range(2):self.slots.append(self._capture_prefix(prepared,noise))
                self.kv_addresses=[tuple(x.data_ptr() for x in runtime.cache_tensors(s.prefix.cache)) for s in self.slots]
                if set(self.kv_addresses[0])&set(self.kv_addresses[1]):raise RuntimeError('KV alias')
                self.shared_scratch=None
                for index in range(2):self._capture_ae(index)
                pools=[g.pool() for s in self.slots for g in (s.vision_graph,s.prefix_graph,s.ae_graph)]
                if len(set(pools))!=len(pools):raise RuntimeError('Concurrent Graph pool alias')
                self.setup_ms=(time.perf_counter()-begin)*1000
            except BaseException:
                self.close()
                raise

        def encode(self,index,prepared,*,dependency=None,metadata_event=None):
            if isinstance(dependency,(tuple,list)) and len(dependency)==3:
                current=self.slots[1-index]
                dependency=(dependency[0],dependency[1],current.iteration_marks[encoder_after_step])
            return super().encode(index,prepared,dependency=dependency,metadata_event=metadata_event)

        def close(self):
            super().close()
            self.shared_scratch=None
            self.policy_constants.clear()
            gc.collect()
            if self.resource_pair is not None:
                # Caller-owned prepared/noise tensors can still have allocator
                # record_stream references. Defer context destruction until
                # bench/serve has returned and those locals have been released.
                pending.append(self.resource_pair)
                self.resource_pair=None

    runtime.ProgressiveGraphs=AdaptedGraphs
    return layouts,pending


def main():
    original_argv=list(sys.argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ae-sm-min',type=int,choices=(0,16,32),default=0)
    parser.add_argument('--encoder-after-step',type=int,choices=(1,4,7,10),default=1)
    parser.add_argument('--serve',action='store_true')
    adapter_args,rest=parser.parse_known_args()
    layouts,pending=install(adapter_args.ae_sm_min,adapter_args.encoder_after_step)
    sys.argv=[sys.argv[0],*rest]
    cleanup=dict(passed=False,pairs_created=len(layouts),contexts_closed=0)
    def finish_contexts():
        import torch
        gc.collect()
        # Drain ALL pairs before destroying any context with allocator events.
        for pair in pending:
            for stream in pair.streams:stream.synchronize()
        if pending:
            with torch.cuda.device(pending[0].device):torch.cuda.empty_cache()
        gc.collect()
        for pair in pending:
            pair.close()
            cleanup['contexts_closed']+=1
        pending.clear()
        gc.collect()
        cleanup.update(passed=True,pairs_created=len(layouts))
    if adapter_args.serve:
        from preparations.idea2.progressive_kv_policy import main as serve, ProgressivePolicy
        original_protocol=ProgressivePolicy.protocol
        ProgressivePolicy.protocol=property(lambda self:dict(original_protocol.fget(self),
            resource_adapter=dict(ae_sm_min=adapter_args.ae_sm_min,encoder_after_step=adapter_args.encoder_after_step,
                                  actual_resources=layouts)))
        try:
            serve()
        except BaseException:
            cleanup['retained_due_to_exception']=True
            raise
        else:
            finish_contexts()
        return
    from results.idea2.progressive_kv_20261025.performance_bench import main as bench
    try:
        bench()
    except BaseException:
        cleanup['retained_due_to_exception']=True
        raise
    else:
        finish_contexts()
    finally:
        if '--output' in rest:
            path=Path(rest[rest.index('--output')+1])/'result.json'
            if path.exists():
                result=json.loads(path.read_text())
                result['resource_adapter']=dict(ae_sm_min=adapter_args.ae_sm_min,
                    encoder_after_step=adapter_args.encoder_after_step,
                    actual_resources=layouts,
                    cleanup=cleanup,
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    command=shlex.join([sys.executable,'-m','results.idea2.progressive_kv_20261025.resource_adapter',*original_argv[1:]]),
                    resource_scope='AE restricted, VLM and Encoder share disjoint remainder; no dynamic quota resize')
                result['command']=result['resource_adapter']['command']
                path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
