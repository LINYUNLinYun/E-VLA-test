import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from . import durable_launcher as launcher

MODULE=__spec__.name


class Durable(unittest.TestCase):
    def test_live_module_and_occupied_port_refuse_launch(self):
        with patch.object(launcher,'instances',return_value=[dict(pid=123)]),self.assertRaisesRegex(RuntimeError,'already alive'):
            launcher.check_empty('target',12345)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
            with patch.object(launcher,'instances',return_value=[]),self.assertRaisesRegex(RuntimeError,'occupied'):
                launcher.check_empty('target',port)

    def test_detached_cpu_child_receipt_and_no_duplicate(self):
        with tempfile.TemporaryDirectory(prefix='durable-cpu-test-') as temporary:
            directory=Path(temporary)
            with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
            # This child only sleeps and prints; no GPU/model/environment import.
            with patch.object(launcher,'instances',return_value=[]):
                receipt=launcher.launch(sys.executable,MODULE,['--cpu-child'],Path.cwd(),port,directory/'launch.json',directory/'run.log')
            pid=receipt['child_pid']
            try:
                current=launcher.identity(pid);self.assertEqual(current['session_id'],pid)
                self.assertEqual(current['start_ticks'],receipt['child']['start_ticks'])
                self.assertIn(MODULE,current['argv']);self.assertEqual(current['cwd'],str(Path.cwd()))
                self.assertEqual(json.loads((directory/'launch.json').read_text())['child_pid'],pid)
                self.assertTrue(any(row['pid']==pid for row in launcher.instances(MODULE)))
                with self.assertRaisesRegex(RuntimeError,'already alive'):launcher.check_empty(MODULE,port)
                with self.assertRaises(FileExistsError):launcher.launch(sys.executable,MODULE,[],Path.cwd(),port,directory/'launch.json',directory/'run.log')
            finally:
                # Exact test-owned CPU process, never a model or renderer.
                current=launcher.identity(pid)
                if current and current['start_ticks']==receipt['child']['start_ticks']:os.kill(pid,15)
                os.waitpid(pid,0)


if __name__=='__main__':
    if '--cpu-child' in sys.argv:
        print('cpu-only detached fixture',flush=True);time.sleep(20)
    else:unittest.main()
