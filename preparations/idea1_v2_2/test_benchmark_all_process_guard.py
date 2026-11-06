import ctypes as C
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import benchmark_all_process_guard as guard


class GuardTests(unittest.TestCase):
    def row(self, start=1., end=1.01):
        return dict(uuid='GPU-test', processes={17:123}, graphics={}, monotonic_start=start, monotonic_end=end)

    def test_local_header_c_layout_matches_ctypes(self):
        if not shutil.which('cc'):
            self.skipTest('C compiler unavailable, ABI must be reviewed separately')
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/'abi'
            source = '#include <stdio.h>\n#include <stddef.h>\n#include "'+guard.HEADER+'"\nint main(){printf("%zu %zu %zu %zu %zu",sizeof(nvmlProcessInfo_v2_t),offsetof(nvmlProcessInfo_v2_t,pid),offsetof(nvmlProcessInfo_v2_t,usedGpuMemory),offsetof(nvmlProcessInfo_v2_t,gpuInstanceId),offsetof(nvmlProcessInfo_v2_t,computeInstanceId));}\n'
            subprocess.run(['cc', '-x', 'c', '-', '-o', str(binary)], input=source.encode(), check=True, capture_output=True)
            actual = list(map(int, subprocess.check_output([str(binary)]).split()))
            expected = [C.sizeof(guard.ProcessInfo)]+[getattr(guard.ProcessInfo,name).offset for name,_ in guard.ProcessInfo._fields_]
            self.assertEqual(actual, expected)
            self.assertEqual(guard.abi_identity()['structure_size'], 24)

    def test_graphics_compute_uuid_and_gap_fail_closed(self):
        guard.validate_sample(self.row(), 'GPU-test', 17)
        for changes in ({'graphics':{99:1}}, {'processes':{}}, {'processes':{17:1,18:1}}, {'uuid':'other'},
                        {'monotonic_end':2.}, {'monotonic_end':float('nan')}, {'monotonic_start':float('inf')},
                        {'monotonic_end':0.}, {'monotonic_start':'invalid'}):
            with self.assertRaises(RuntimeError): guard.validate_sample(dict(self.row(), **changes), 'GPU-test', 17)
        with self.assertRaisesRegex(RuntimeError, 'gap'):
            guard.validate_sample(self.row(2.,2.01), 'GPU-test',17,self.row())

    def test_process_list_capacity_retry_and_unsupported(self):
        calls = []
        def fake(handle, count, values):
            calls.append(1)
            if len(calls)==1:
                count._obj.value=20
                return 7
            count._obj.value=1
            values[0].pid=17
            values[0].usedGpuMemory=128*1024**2
            return 0
        self.assertEqual(guard.Scanner.processes(fake,None), {17:128})
        with self.assertRaises(RuntimeError): guard.Scanner.processes(lambda *args:3,None)
        def duplicate(handle,count,values):
            count._obj.value=2
            values[0].pid=values[1].pid=17
            return 0
        with self.assertRaises(RuntimeError): guard.Scanner.processes(duplicate,None)

    def test_monitor_retains_reappearing_graphics_and_never_recovers_to_pass(self):
        local=dict(pid=1,start_ticks='start',pid_namespace='ns')
        rows=[dict(self.row(1,1.01)),dict(self.row(1.1,1.11),graphics={99:1})]
        class FakeScanner:
            def __init__(self,**kwargs): pass
            def scan(self): return rows.pop(0)
            def close(self): pass
        class Pipe:
            requests=[('check','sample:before'),('flush','point00'),('stop','final')]
            sent=[]
            first=True
            def poll(self,timeout):
                if self.first:
                    self.first=False;return False
                return bool(self.requests)
            def recv(self): return self.requests.pop(0)
            def send(self,value): self.sent.append(value)
        pipe=Pipe()
        with patch.object(guard,'Scanner',FakeScanner),patch.object(guard,'parent_identity',return_value=local):
            guard.monitor_child(pipe,'GPU-test',17,local)
        self.assertIsNone(pipe.sent[0]['error'])
        self.assertTrue(all(row['error'] for row in pipe.sent[1:]))
        self.assertEqual(pipe.sent[2]['events'][-1]['graphics'],{99:1})
        self.assertEqual(pipe.sent[2]['events'][-1]['label'],'periodic')

    def test_monitor_dead_and_timeout_raise_before_forward(self):
        monitor=guard.Monitor.__new__(guard.Monitor)
        class Process:
            def is_alive(self):return False
        monitor.process=Process()
        with self.assertRaisesRegex(RuntimeError,'died'): monitor.check('before')
        class Pipe:
            def poll(self,timeout):return False
        monitor.pipe=Pipe()
        with self.assertRaisesRegex(RuntimeError,'unresponsive'): monitor.receive()

    def test_parent_identity_drift_and_graphics_api_error_are_monitor_failures(self):
        local=dict(pid=1,start_ticks='start',pid_namespace='ns')
        for mode in ('identity','api'):
            class Scanner:
                def __init__(self,**kwargs):pass
                def scan(self):raise RuntimeError('graphics API unsupported')
                def close(self):pass
            class Pipe:
                def __init__(self):self.sent=[]
                def poll(self,timeout):return True
                def recv(self):return ('stop','final')
                def send(self,value):self.sent.append(value)
            pipe=Pipe()
            with patch.object(guard,'Scanner',Scanner),patch.object(guard,'parent_identity',return_value={} if mode=='identity' else local):
                guard.monitor_child(pipe,'GPU-test',17,local)
            self.assertTrue(all(row['error'] for row in pipe.sent))


if __name__ == '__main__': unittest.main()
