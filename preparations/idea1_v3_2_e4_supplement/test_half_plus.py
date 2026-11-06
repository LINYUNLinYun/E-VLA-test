from collections import Counter
import json
import unittest
from .half_plus import subset,modes
from .native_envs import ROOT


class HalfSelectionTests(unittest.TestCase):
    def test_outcome_blind_balanced_half(self):
        path=ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_20261022T074328Z/manifest.json'
        selection=json.loads(path.read_text())['case_selection'];indices,original,repairs=subset(selection)
        cases=selection['selected'];self.assertEqual(len(indices),175);self.assertEqual(len(original),20)
        self.assertEqual(len({(cases[i]['mother_suite'],cases[i]['mother_task']) for i in indices}),40)
        self.assertEqual(set(Counter((cases[i]['category'],cases[i]['difficulty_level']) for i in indices).values()),{5})
        self.assertEqual(set(Counter(i//10 for i in original).values()),{5})
        self.assertTrue({0,10,20,30}.issubset(original))
        for repair in repairs:
            a,b=cases[repair['remove_source_index']],cases[repair['add_source_index']]
            self.assertEqual((a['category'],a['difficulty_level'],a['mother_suite']),(b['category'],b['difficulty_level'],b['mother_suite']))
        self.assertTrue(all(len(modes(cases[i]))==3 and modes(cases[i])[-1]==cases[i]['matching_draft'] for i in indices))


if __name__=='__main__': unittest.main()
