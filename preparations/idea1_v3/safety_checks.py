"""CPU-only recovery-gate and model-child dispatch checks; no GPU or subprocess."""
import json
from pathlib import Path
import subprocess
import tempfile
import types
from unittest.mock import patch

from preparations.idea1_v3.safety_runtime import kernel_digest,require_recovery_pass
from preparations.idea1_v3.safe_worker import run


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); wd=root/'workers/worker0'; wd.mkdir(parents=True)
        row=dict(key='test-recovered-key',experiment='E1',condition='test',execution='A',suite='libero_goal',task=8,init=7,owner=0)
        (root/'manifest.json').write_text(json.dumps(dict(tasks=[row])))
        complete=root/'episodes/E1/test/A/libero_goal/task08/init07/complete.json'; complete.parent.mkdir(parents=True)
        complete.write_text(json.dumps(dict(key=row['key'],valid=True)))
        receipt=wd/'reference.json'; receipt.write_text(json.dumps(dict(passed=True,source_sha256=kernel_digest())))
        guard=wd/'guard.json'; record=dict(key=row['key'],status='valid',revision='qk-row-bounds-v1',
            source_sha256={'safe_qk.py':kernel_digest()},reference_receipt=str(receipt))
        guard.write_text(json.dumps(record)); assert require_recovery_pass(guard)['key']==row['key']
        record['status']='reserved'; guard.write_text(json.dumps(record))
        try: require_recovery_pass(guard)
        except RuntimeError: pass
        else: raise AssertionError('uncompleted single-key gate must block')
        record['status']='valid'; guard.write_text(json.dumps(record))
        args=types.SimpleNamespace(attempt=root,shard=0,gpu=2,safety_gate=guard)
        captured=[]
        def capture(command,*args,**kwargs): captured.append((command,kwargs)); return object()
        for stage,base,target in (('E1','preparations.idea1_v3.server','preparations.idea1_v3.cli.worker'),
                                  ('E3','preparations.idea1_v3.branch_server','preparations.idea1_v3.branches.worker')):
            def worker(_args): subprocess.Popen(['python','-m',base,'--config','{}'],env={})
            with patch(target,side_effect=worker),patch('subprocess.Popen',side_effect=capture):
                run(args,dict(stage=stage,shards=1,tasks=[row]))
            command,kwargs=captured[-1]
            assert command[2]=='preparations.idea1_v3.safe_worker_server'
            assert kwargs['env']['V3_SAFE_BASE_SERVER']==base
            assert kwargs['env']['V3_QK_WORKER_RECEIPT'].endswith('.json')
        # A technical invalid without a completion is never silently retried.
        (wd/'events.jsonl').write_text(json.dumps(dict(kind='invalid',key=row['key']))+'\n')
        complete.unlink()
        try: run(args,dict(stage='E1',shards=1,tasks=[row]))
        except FileNotFoundError: pass  # Recovery completion gate is checked first.
        else: raise AssertionError('missing valid recovery completion must block')
    print('CPU safety gate rejects unfinished recovery; E1/E3 child-server rewrite+receipt survives dispatch; no GPU/subprocess executed: OK')

if __name__=='__main__': main()
