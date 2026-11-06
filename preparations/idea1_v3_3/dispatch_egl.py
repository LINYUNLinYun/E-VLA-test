"""Frozen dispatch ownership, correct env-only EGL binding, same model CUDA."""
from . import dispatch
from .egl_mapping import worker_environment
original=dispatch.subprocess.Popen


def popen(command,*args,**kwargs):
    command=list(command)
    if 'preparations.idea1_v3_3.server' in command and command[command.index('--model')+1]=='pi0':
        command[command.index('preparations.idea1_v3_3.server')]='preparations.idea1_v3_3.server_r2'
    if 'preparations.idea1_v3_3.rollout' in command:
        env=kwargs['env']; physical=int(env['CUDA_VISIBLE_DEVICES'])
        kwargs['env']=worker_environment(env,physical)
        command[command.index('preparations.idea1_v3_3.rollout')]='preparations.idea1_v3_3.rollout_egl'
    return original(command,*args,**kwargs)


dispatch.subprocess.Popen=popen
if __name__=='__main__': dispatch.main()
