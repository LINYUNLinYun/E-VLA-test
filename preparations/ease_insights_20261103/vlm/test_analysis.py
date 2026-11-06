"""Selection tests using synthetic outcomes, not experimental evidence."""
import tempfile
import unittest
from pathlib import Path
from .analyze import exact_paired, expected_cases, freeze, key, select_pair
from .common import SUITES, conditions


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(conditions=conditions(), suites=list(SUITES), tasks=list(range(10)), discovery_inits=[30])
        self.ids = [c["id"] for c in self.manifest["conditions"]]
        self.costs = dict(valid_services=1, mean_cycle_ms={c: 30 if c == "full_d10" else 20 for c in self.ids},
            faster_than_full={c: True for c in self.ids[1:]})

    def records(self, rates):
        return {(cid,key(case)): dict(success=case["task"] < rates[cid][self.manifest["suites"].index(case["suite"])])
            for cid in self.ids for case in expected_cases(self.manifest)}

    def test_maximin_crossing_and_fixed_ties(self):
        full,a,b,c = self.ids
        records = self.records({full:[9]*4,a:[8,5,8,5],b:[5,8,5,8],c:[5,8,5,8]})
        result = select_pair(self.manifest,records,self.costs,True)
        self.assertEqual(result["status"],"selected")
        self.assertEqual(result["selected"]["pair"],[a,b])
        self.assertEqual([d["suite"] for d in result["selected"]["directions"]],list(SUITES[:2]))
        self.assertAlmostEqual(result["selected"]["objective_min_opposite_suite_difference"],.3)

    def test_cost_and_overall_quality_are_hard_gates(self):
        full,a,b,c = self.ids
        records = self.records({full:[9]*4,a:[8,5,6,6],b:[5,8,6,6],c:[0]*4})
        self.costs["faster_than_full"][b] = False
        self.assertEqual(select_pair(self.manifest,records,self.costs,True)["status"],"none")
        self.costs["faster_than_full"][b] = True
        records = self.records({full:[9]*4,a:[8,5,6,6],b:[5,8,5,5],c:[0]*4})
        self.assertEqual(select_pair(self.manifest,records,self.costs,True)["status"],"none")

    def test_universal_winner_does_not_imply_complementarity(self):
        full,a,b,c = self.ids
        records = self.records({full:[9]*4,a:[9]*4,b:[7]*4,c:[6]*4})
        self.assertEqual(select_pair(self.manifest,records,self.costs,True)["status"],"none")

    def test_incomplete_cannot_select_or_freeze(self):
        self.assertEqual(select_pair(self.manifest,{},self.costs,False)["status"],"incomplete_discovery")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze(Path(directory),dict(complete=True,valid_episodes=159,expected_episodes=160,selection=dict(status="none")))
            self.assertFalse((Path(directory)/"selection.json").exists())

    def test_exact_discordant_probability(self):
        self.assertEqual(exact_paired(0,0),1)
        self.assertEqual(exact_paired(3,0),.25)
        self.assertEqual(exact_paired(3,0,one_sided=True),.125)
        self.assertEqual(exact_paired(3,1),.625)


if __name__ == "__main__": unittest.main()
