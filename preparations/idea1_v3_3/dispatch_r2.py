"""Reuse frozen condition ownership/dispatch; select metadata successor server."""
from . import dispatch
original=dispatch.subprocess.Popen


def popen(command,*args,**kwargs):
    command=list(command)
    if 'preparations.idea1_v3_3.server' in command:
        command[command.index('preparations.idea1_v3_3.server')]='preparations.idea1_v3_3.server_r2'
    return original(command,*args,**kwargs)


dispatch.subprocess.Popen=popen
if __name__=='__main__': dispatch.main()
