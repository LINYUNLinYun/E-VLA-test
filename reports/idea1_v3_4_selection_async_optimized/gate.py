import argparse
from pathlib import Path
from preparations.idea1_v3_4_selection_async.gate import run as original_gate
from reports.idea1_v3_4_selection_async_optimized_gate import run as optimized_gate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();original_gate(a.port,a.output);optimized_gate(a.port,Path(a.output)/'optimized')
