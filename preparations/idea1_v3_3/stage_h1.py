"""Resume the original frozen shard with metadata-only probe successor."""
import argparse
from . import stage

original=stage.subprocess.Popen


def launch(command,*args,**kwargs):
    command=list(command)
    if 'preparations.idea1_v3_3.server_c1' in command:
        command[command.index('preparations.idea1_v3_3.server_c1')]='preparations.idea1_v3_3.server_c1_h1'
    return original(command,*args,**kwargs)


stage.subprocess.Popen=launch
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);a=p.parse_args()
    stage.run(a.manifest,a.gpu,a.model,a.shard,a.shards)
