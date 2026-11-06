import unittest
from unittest.mock import patch
import torch
from .policy_calvin_server import final_allocation_guard


class AllocationGuard(unittest.TestCase):
    def test_guard_once_immediately_before_first_cuda_move_without_real_gpu(self):
        events=[]
        def moved(module,*args,**kwargs):events.append(("to",args,kwargs));return module
        def idle():events.append(("idle",));return {"preexisting_compute_processes":""}
        module=torch.nn.Linear(1,1)
        with patch.object(torch.nn.Module,"to",moved),patch("preparations.idea1_v3_2.residual.require_idle_gpu",idle):
            with final_allocation_guard() as receipt:
                module.to(torch.bfloat16)
                module.to("cuda")
                module.to(device="cuda")
            self.assertIs(torch.nn.Module.to,moved)
        self.assertEqual([e[0] for e in events],["to","idle","to","to"])
        self.assertTrue(receipt["checked"])

    def test_foreign_allocation_stops_before_any_cuda_move(self):
        events=[]
        def moved(module,*args,**kwargs):events.append("cuda_move");return module
        module=torch.nn.Linear(1,1)
        with patch.object(torch.nn.Module,"to",moved),patch("preparations.idea1_v3_2.residual.require_idle_gpu",side_effect=RuntimeError("GPU_NOT_IDLE")):
            with self.assertRaisesRegex(RuntimeError,"GPU_NOT_IDLE"):
                with final_allocation_guard():module.to("cuda")
        self.assertEqual(events,[])


if __name__=="__main__":unittest.main()
