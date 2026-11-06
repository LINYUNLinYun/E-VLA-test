from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import supervise_plus_v14_d2 as module


class Supervisor(unittest.TestCase):
    def test_exact_service_identity_required(self):
        args=SimpleNamespace(port=1,expected_server_pid=22,gpu_uuid='assigned')
        health=dict(manifest_hash='fixed',server_pid=22,gpu_uuid='assigned',modes=['v14_d2'],D=2,legacy_gate_passed=True)
        with patch.object(module,'rpc',return_value=dict(protocol=health)):
            self.assertEqual(module.assert_service(args,'fixed'),health)
        for key,value in [('server_pid',23),('gpu_uuid','other'),('manifest_hash','other'),('D',10),('legacy_gate_passed',False)]:
            changed=dict(health);changed[key]=value
            with patch.object(module,'rpc',return_value=dict(protocol=changed)),self.assertRaises(RuntimeError):module.assert_service(args,'fixed')


if __name__=='__main__':unittest.main()
