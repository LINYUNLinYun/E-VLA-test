import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from .analyze_plus_half import matched_jobs,summaries,common_distance,completion_gate,comparison_labels


class HalfAnalysisTest(unittest.TestCase):
    def fixture(self):
        jobs=[dict(group='perturbed350',source_index=i,case_hash=str(i),modes=['full_d10','v14_d10','draft_goal'],case=dict(mother_suite='libero_goal',category='Lighting',difficulty_level=1)) for i in range(2)]
        rows=[dict(group='perturbed350',source_index=i,mode=m,gpu_uuid='same',success=(m!='full_d10')) for i in range(2) for m in jobs[i]['modes'] if not (i==1 and m=='draft_goal')]
        return dict(jobs=jobs),dict(selected_completed=rows,extra_historical_completed=[dict(source_index=999)])

    def test_only_three_way_common_cases(self):
        m,l=self.fixture();paired=matched_jobs(m,l)
        self.assertEqual(len(paired),1);self.assertEqual(paired[0][0]['source_index'],0)
        rows=summaries(m,paired)
        rs=[r for r in rows if r['group']=='perturbed350' and r['dimension']=='overall']
        self.assertEqual({r['paired_cases'] for r in rs},{1})
        self.assertEqual(rs[1]['paired_difference_vs_full'],1)
        self.assertFalse(any(r['paired_cases'] for r in rows if r['group']=='original40'))

    def test_final_requires_all585_and195(self):
        for a,b in [(584,195),(585,194),(100,20)]:
            with self.assertRaises(RuntimeError):completion_gate(a,b,False)
            self.assertFalse(completion_gate(a,b,True))
        self.assertTrue(completion_gate(585,195,False))

    def test_distance_stops_at_common_time_no_padding(self):
        t,d=common_distance(np.array([0,1,2]),np.array([[0,0,0],[1,0,0],[2,0,0]]),np.array([0,.5]),np.array([[0,0,0],[0,0,0]]))
        self.assertEqual(t.tolist(),[0,.5]);self.assertEqual(d.tolist(),[0,500])

    def test_gpu_and_duplicate_identity_rejected(self):
        m,l=self.fixture();l['selected_completed'][1]['gpu_uuid']='other'
        with self.assertRaises(ValueError):matched_jobs(m,l)
        m,l=self.fixture();l['selected_completed'].append(l['selected_completed'][0])
        with self.assertRaises(ValueError):matched_jobs(m,l)

    def test_reader_labels_include_actual_outcome_steps_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            rows={}
            for mode,success,steps in [('full_d10',True,93),('v14_d10',False,600)]:
                path=Path(directory)/(mode+'.json');path.write_text(json.dumps(dict(success=success,steps=steps,native_info=dict(instruction='put bowl on plate'))));rows[mode]=dict(path=str(path))
            labels,instruction=comparison_labels(rows,'v14_d10')
            self.assertEqual(instruction,'put bowl on plate')
            self.assertEqual(labels['full_d10'],'Full: success, 93 steps')
            self.assertEqual(labels['v14_d10'],'V14: failure, 600 steps')


if __name__=='__main__':unittest.main()
