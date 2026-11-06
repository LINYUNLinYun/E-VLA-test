"""Safe E1 expansion worker capturing existing init7/task0 requests only.

Same worker arguments as e1_cli worker. No extra episode or prediction.
"""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess


def main():
    from preparations.idea1_v3 import episode
    from preparations.idea1_v3_supplement import e1_cli
    original_run=episode.run; original_rpc=episode.rpc; original_spawn=subprocess.run
    active=None
    def rpc(port,request):
        if active is not None and request.get('op')=='predict' and len(active)<25:
            # Serialize before transport so neither downstream mutation nor GPU
            # output can become part of the saved real input.
            active.append(pickle.loads(pickle.dumps({k:v for k,v in request.items() if k!='op'},protocol=5)))
        return original_rpc(port,request)
    def run(directory,port,row,*args,**kwargs):
        nonlocal active
        selected=row['task']==0 and row['init']==7
        active=[] if selected else None
        try:
            result=original_run(directory,port,row,*args,**kwargs)
            if selected:
                path=Path(directory)/'real_predict_inputs_first25.pkl'
                if path.exists(): raise FileExistsError('Never overwrite captured inputs')
                raw=pickle.dumps(active,protocol=5); path.write_bytes(raw)
                from preparations.idea1_v3.io import atomic
                atomic(path.with_suffix('.json'),dict(key=row['key'],suite=row['suite'],task=0,init=7,
                    requests=len(active),sha256=hashlib.sha256(raw).hexdigest(),source='actual existing closed-loop predict request payloads; max25, no extra prediction',
                    benchmark_boundary='future replay acknowledgements are virtual, not this physical history'))
            return result
        finally: active=None
    def spawn(command,*args,**kwargs):
        command=list(command)
        if 'preparations.idea1_v3_supplement.e1_cli' in command:
            command[command.index('preparations.idea1_v3_supplement.e1_cli')]='preparations.idea1_v3_supplement.capture_worker'
        return original_spawn(command,*args,**kwargs)
    episode.run=run; episode.rpc=rpc; subprocess.run=spawn
    try: e1_cli.main()
    finally: episode.run=original_run; episode.rpc=original_rpc; subprocess.run=original_spawn


if __name__=='__main__': main()
