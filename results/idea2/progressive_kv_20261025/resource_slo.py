"""Absolute camera/KV-age sampling for the selected resource candidate.

No local unrestricted baseline is included; do not assign a cross-GPU SLO
budget pass by comparing these samples with another physical GPU's results.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import sys


def main():
    original=list(sys.argv)
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--ae-sm-min',type=int,choices=(16,32),required=True)
    options,rest=p.parse_known_args()
    from results.idea2.progressive_kv_20261025.resource_adapter import install
    layouts,pending=install(options.ae_sm_min,1)
    sys.argv=[sys.argv[0],*rest]
    from results.idea2.progressive_kv_20261025.slo_bench import main as sample
    cleanup=False
    try:
        sample()
        import torch
        gc.collect()
        for pair in pending:
            for stream in pair.streams:stream.synchronize()
        if pending:
            with torch.cuda.device(pending[0].device):torch.cuda.empty_cache()
        gc.collect()
        for pair in pending:pair.close()
        pending.clear()
        cleanup=True
    finally:
        if '--output' in rest:
            path=Path(rest[rest.index('--output')+1])/'result.json'
            if path.exists():
                result=json.loads(path.read_text())
                result['resource_adapter']=dict(ae_sm_min=options.ae_sm_min,encoder_after_step=1,
                    actual_resources=layouts,cleanup=dict(passed=cleanup),
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                result['command']=shlex.join([sys.executable,'-m','results.idea2.progressive_kv_20261025.resource_slo',*original[1:]])
                result['slo_comparison_scope']='Absolute candidate ages only; no unrestricted same-GPU baseline and no cross-GPU budget decision'
                if not cleanup:
                    result.update(status='failed_cleanup',performance_eligible=False)
                path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
