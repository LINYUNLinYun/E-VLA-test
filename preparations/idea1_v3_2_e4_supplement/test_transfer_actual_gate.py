import unittest
from .transfer_actual_gate import physical_probes,selected_cases


class ActualGateTest(unittest.TestCase):
    def test_probe_signs_and_units(self):
        probes=physical_probes('calvin');self.assertEqual(len(probes),14)
        for axis,sign,action in probes:self.assertEqual(action[axis],sign*(1 if axis==6 else .2))
        self.assertEqual(len(physical_probes('pusht')),4)

    def test_selected_official_cases(self):
        cases=[dict(environment='simpler',task=t,index=i) for t in ('google_robot_pick_coke_can','google_robot_move_near') for i in range(50)]
        self.assertEqual([c['index'] for c in selected_cases(dict(cases=cases),'simpler')],[0,24,25,49,0,49])


if __name__=='__main__':unittest.main()
