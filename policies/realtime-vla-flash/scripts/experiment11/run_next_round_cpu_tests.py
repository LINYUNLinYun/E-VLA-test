"""Persist reproducible CPU test evidence for the final handoff."""
import importlib
import io
import json
from pathlib import Path
import time
import unittest
from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json, sha256

MODULES = ('test_next_round_audit','test_task_type_probe_analysis','test_task_type_probe',
           'test_probe_parity_diagnostic','test_call_budget','test_shadow_risk_metrics','test_failure_events')
PRODUCTION = ('audit_next_round','analyze_task_type_probe','run_task_type_layer_probe',
              'diagnose_probe_parity','replay_state','call_budget','shadow_risk_metrics','analyze_failure_events')
PRODUCTION = (*PRODUCTION,'account_probe_internal_work')


def main():
    modules = ['scripts.experiment11.'+name for name in MODULES]
    stream = io.StringIO();started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    hashes = {}
    for name in (*MODULES,*PRODUCTION):
        module = importlib.import_module('scripts.experiment11.'+name)
        path = Path(module.__file__);hashes[str(path)] = sha256(path)
    output = RESULTS/'analysis/next_round/round2_20261011/cpu_tests.json'
    record = {'complete':result.wasSuccessful() and not result.skipped,'tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
              'modules':modules,'started_epoch':started,'completed_epoch':time.time(),
              'source_hashes':hashes,'transcript':stream.getvalue(),
              'scope':'CPU protocol/statistics/accounting tests;not a substitute for full raw data/GPU parity/video audit'}
    dump_json(output,record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('transcript','source_hashes')}))
    return 0 if record['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
