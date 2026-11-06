"""Frozen-runtime capacity harness including CPU action decode in formal timing."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time


def install_decode():
    import numpy as np
    from preparations.idea2 import progressive_kv as runtime
    Base = runtime.ProgressiveRuntime

    class DecodedRuntime(Base):
        def request(self, prepared, noise, **kwargs):
            output,row,nxt=super().request(prepared,noise,**kwargs)
            begin=time.perf_counter()
            state=prepared.state[0].detach().cpu().numpy()
            decoded=self.backend.policy._output_transform(dict(state=state,actions=output.copy()))['actions']
            if np.asarray(decoded).shape!=(10,7) or not np.isfinite(decoded).all():
                raise RuntimeError('Invalid decoded action chunk')
            elapsed=(time.perf_counter()-begin)*1000
            row['normalized_runtime_host_ms']=row['cpu_cpu_ms']
            row['cpu_cpu_ms']+=elapsed
            row['decode_host_ms']=elapsed
            row['timing_scope']='Prepared runtime input to decoded CPU action; input admission is separate'
            return output,row,nxt
    runtime.ProgressiveRuntime=DecodedRuntime


def main():
    command=shlex.join([sys.executable,'-m','results.idea2.progressive_kv_20261025.performance_bench',*sys.argv[1:]])
    install_decode()
    from preparations.idea2.progressive_kv_bench import main as bench
    try:
        bench()
    finally:
        if '--output' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--output')+1])/'result.json'
            if p.exists():
                d=json.loads(p.read_text());d['command']=command
                d['performance_driver']=dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    decoded_cpu_output_included=True,
                    admission_field_correction='Base admission_to_action_ms is loop-entry to decoded return; never true observation admission. Rename in final analysis.')
                p.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
