"""Synthetic CPU interface checks; these are not experimental measurements."""
import copy
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load('fresh_analysis_cpu', HERE / 'analyze_traces.py')
fixture_module = load('legacy_fixture_cpu', analyzer.LEGACY_PATH.parent / 'analysis_selftest/test_analyzer.py')


class FreshTailChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sql = self.root / 'synthetic.sqlite'
        self.profile = fixture_module.fixture(self.sql)
        with sqlite3.connect(self.sql) as db:
            db.execute('create table TARGET_INFO_CUDA_CONTEXT_INFO(processId integer,deviceId integer,contextId integer,parentContextId integer,isGreenContext integer,numMultiprocessors integer)')
            db.executemany('insert into TARGET_INFO_CUDA_CONTEXT_INFO values(?,?,?,?,?,?)',
                           [(1, 0, 1, 0, 0, 0), (1, 0, 2, 1, 1, 32), (1, 0, 3, 1, 1, 96)])
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set greenContextId=case when streamId=3 then 2 else 3 end')
            db.execute('delete from CUPTI_ACTIVITY_KIND_CUDA_EVENT')
        self.profile['fresh_tail_adapter'] = dict(ae_sm_min=32, fresh_tail_steps=1,
            actual_resources=[dict(sm_counts=[32, 96], device_sm_count=128)],
            sha256=analyzer.digest(HERE / 'adapter.py'), cleanup=dict(passed=True))
        for row in self.profile['rows']:
            row.update(fresh_tail_steps=1, frontiers=[3] * 9 + [18], fresh_tail_wait_ms=3.4)
        self.before = analyzer.digest(analyzer.LEGACY_PATH)

    def tearDown(self):
        self.assertEqual(self.before, analyzer.digest(analyzer.LEGACY_PATH))
        self.temp.cleanup()

    def run_analysis(self):
        return analyzer.analyze(self.sql, self.root / 'output', self.profile,
                                dict(synthetic=True, gpu_launched=False))

    def test_fallback_and_no_host_clipping_or_invented_wait(self):
        original = copy.deepcopy(self.profile)
        report = self.run_analysis()
        audit = report['resource_audit']
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['adapter_metadata_source'], 'fresh_tail_adapter')
        self.assertEqual(audit['expected_sm_pair'], [32, 96])
        self.assertEqual({r['kernel_primary_context_id'] for r in audit['assignments']}, {1})
        first = report['requests'][0]
        self.assertEqual(first['AE_x_currentVLM_ms'], 20)
        self.assertEqual(first['AE_x_nextVision_ms'], 15)
        self.assertGreater(first['graph_kernels_after_host_NVTX'], 0)
        self.assertEqual(first['fresh_tail_wait_ms_runtime_event'], 3.4)
        self.assertFalse(report['fresh_tail_analysis']['nsys_wait_timestamps_inferred'])
        self.assertIsNone(report['crosschecks'][0].get('next_encoder_finish_minus_AE_finish_nsys_event_ms'))
        self.assertEqual(original, self.profile)
        self.assertIs(analyzer.legacy.resource_audit, analyzer._resource_audit)

    def test_missing_runtime_wait_is_rejected(self):
        del self.profile['rows'][0]['fresh_tail_wait_ms']
        with self.assertRaisesRegex(ValueError, 'Missing runtime'):
            self.run_analysis()

    def test_incomplete_mandatory_tail_rejected(self):
        self.profile['rows'][0]['frontiers'][-1] = 17
        with self.assertRaisesRegex(ValueError, 'frontier was incomplete'):
            self.run_analysis()

    def test_primary_context_cannot_substitute_for_green_context(self):
        with sqlite3.connect(self.sql) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set greenContextId=1 where streamId=3')
        self.assertFalse(self.run_analysis()['resource_audit']['passed'])

    def test_fresh_snapshot_order_is_independent_of_wait_duration(self):
        with sqlite3.connect(self.sql) as db:
            for corr, base in [(4, 179), (7, 249)]:
                for i in range(17):
                    start = int((base + 0.02 + i * 0.02) * fixture_module.MS)
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                        (start, start + 1000, 0, 1, 3, 2, corr, fixture_module.PID, 3, 121))
            for corr, base in [(5, 162), (8, 232)]:
                for i in range(9):
                    start = int((base + i * 4) * fixture_module.MS)
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?)',
                        (start, start + 1000, 0, 1, 2, 3, corr, fixture_module.PID, 4, 130))
        report = self.run_analysis()
        evidence = report['fresh_tail_analysis']['kernel_ready_ordering']
        self.assertTrue(evidence['passed'])
        self.assertEqual(evidence['rows'][0]['fresh_snapshot_after_last_publish_ms'], [14.659])
        self.assertEqual(report['requests'][0]['fresh_tail_wait_ms_runtime_event'], 3.4)
        with sqlite3.connect(self.sql) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set end=? where correlationId=4 and demangledName=3',
                       (199 * fixture_module.MS,))
        self.assertFalse(self.run_analysis()['fresh_tail_analysis']['kernel_ready_ordering']['passed'])


if __name__ == '__main__':
    unittest.main()
