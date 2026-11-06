"""CPU-only review probes; production files are read-only."""
import json
from pathlib import Path
from preparations.idea2.test_submission import SubmissionTests, Event
from preparations.idea2.cewp_preparation import PreparedModulationCore

out = Path(__file__).parent
fixture = SubmissionTests()
fixture.setUp()
results = {}
try:
    backend, policy = fixture.make('ae_first')
    backend.graph_core_class = PreparedModulationCore
    backend.await_previous_prefix = True
    metadata = policy.protocol
    assert metadata['preparation_core'] == 'PreparedModulationCore'
    assert metadata['prepared_schedule'] is True
    assert metadata['await_previous_prefix'] is True
    del backend.graph_core_class
    del backend.await_previous_prefix
    assert policy.protocol['preparation_core'] is None
    assert policy.protocol['prepared_schedule'] is False
    assert policy.protocol['await_previous_prefix'] is False
    results['protocol_metadata'] = {'passed': True, 'enabled_preparation_core': metadata['preparation_core'], 'disabled_preparation_core': policy.protocol['preparation_core']}
    def fail_output(*args):
        raise ValueError('injected CPU output transform failure')
    backend.output = fail_output
    try:
        policy.predict(1, 'task', 'suite', 2)
    except ValueError as exc:
        results['output_failure'] = {'error': str(exc), 'leases_after_error': [s.leases for s in policy.pool.slots]}
    try:
        policy.reset()
        results['output_failure']['reset_succeeded'] = True
    except RuntimeError as exc:
        results['output_failure'].update(reset_succeeded=False, reset_error=str(exc))
    # Repair only the CPU fake fixture so unittest cleanup can run.
    for slot in policy.pool.slots:
        slot.leases = 0
    policy.reset()
finally:
    fixture.doCleanups()
(out / 'final_review_probes.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
