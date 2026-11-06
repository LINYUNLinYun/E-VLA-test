"""Same scientific queue; actual render binding and per-worker probe evidence."""
from . import dispatch
from .dispatch import OUT
from .egl_mapping import worker_environment
original=dispatch.subprocess.Popen


def popen(command,*args,**kwargs):
    command=list(command)
    if 'preparations.idea1_v3_3.server' in command and command[command.index('--model')+1]=='pi0':
        module='preparations.idea1_v3_3.server_c1' if (OUT/'E2_manifest.json').exists() else 'preparations.idea1_v3_3.server_r2'
        command[command.index('preparations.idea1_v3_3.server')]=module
    if 'preparations.idea1_v3_3.rollout' in command:
        env=kwargs['env'];kwargs['env']=worker_environment(env,int(env['CUDA_VISIBLE_DEVICES']))
        command[command.index('preparations.idea1_v3_3.rollout')]='preparations.idea1_v3_3.rollout_bound'
        if command[command.index('--condition')+1].startswith('E2_'):command[command.index('--manifest')+1]=str(OUT/'E2_manifest.json')
    return original(command,*args,**kwargs)


dispatch.subprocess.Popen=popen
if __name__=='__main__':dispatch.main()
