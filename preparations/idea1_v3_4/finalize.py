"""Observe current handles, then run the frozen experiment's final evidence path."""
import hashlib
import json
import os
import subprocess
import sys
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment,lock
from .manifest import ROOT,OUT


def live(pid,module):
    p=subprocess.run(['ps','-p',str(pid),'-o','args='],text=True,capture_output=True)
    return p.returncode==0 and module in p.stdout


def main():
    with lock(OUT/'finalize.lock',blocking=False):
        state=dict(pid=os.getpid(),started=time.time(),status='waiting',observed_handles={
            'controller':809588,'noise_control_watch':818904})
        atomic(OUT/'finalization_state.json',state)
        while live(809588,'preparations.idea1_v3_4.controller') or live(818904,'preparations.idea1_v3_4.noise_control'):
            time.sleep(30)
        original=json.loads((OUT/'status.json').read_text())
        noise=json.loads((OUT/'E3_noise_control/audit.json').read_text())
        if original['status']!='rollouts_complete' or not noise['passed']:
            raise RuntimeError('Execution did not complete successfully; inspect the real task handles/logs')
        for module,extra in (('analyze',['--plots']),('audit',[]),('report',[])):
            state.update(status='running',stage=module,updated=time.time())
            atomic(OUT/'finalization_state.json',state)
            with (OUT/f'final_{module}.log').open('w') as log:
                p=subprocess.run([sys.executable,'-m','preparations.idea1_v3_4.'+module,*extra],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
            if p.returncode: raise RuntimeError('Final '+module+' failed; inspect final_'+module+'.log')
        state.update(status='artifacts_ready',finished=time.time(),source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
        atomic(OUT/'finalization_state.json',state)
        print('ARTIFACTS_READY; visual and requirement review remains with the goal owner',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        atomic(OUT/'finalization_error.json',dict(time=time.time(),error=repr(exc)))
        raise
