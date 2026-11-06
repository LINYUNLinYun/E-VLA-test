"""Correct EGL dispatch and direct every formal E2 case to its new manifest."""
from . import dispatch_egl,dispatch
from .dispatch import OUT
original=dispatch.subprocess.Popen


def popen(command,*args,**kwargs):
    command=list(command)
    if 'preparations.idea1_v3_3.server' in command and command[command.index('--model')+1]=='pi0':
        command[command.index('preparations.idea1_v3_3.server')]='preparations.idea1_v3_3.server_c1'
    if 'preparations.idea1_v3_3.rollout' in command and command[command.index('--condition')+1].startswith('E2_'):
        command[command.index('--manifest')+1]=str(OUT/'E2_manifest.json')
    return original(command,*args,**kwargs)


dispatch.subprocess.Popen=popen
if __name__=='__main__': dispatch.main()
