import unittest
from scripts.experiment11.analyze_failure_events import chronology


class EventTest(unittest.TestCase):
    def row(self,step,grasp,command=1,height=.02):
        return {'step':step,'when':'post_action','primary_objects':['obj'],
                'robot_grasps':{'obj':grasp},'robot_contacts':{'obj':grasp},
                'action':[0]*6+[command],
                'objects':{'obj':{'type':'object','joint':None,'pos':[0,0,height]}}}

    def test_open_after_stable_grasp_and_loss_confirmation(self):
        rows=[self.row(i,True) for i in range(3)]+[self.row(i,False,-1) for i in range(3,6)]
        summary,events=chronology(rows)
        self.assertEqual(summary['first_open_while_stably_grasped'],3)
        self.assertEqual(summary['first_grasp_loss_candidate'],3)
        self.assertTrue(any(e['kind']=='grasp_loss_candidate' and e['confirmed_at_step']==5 for e in events))

    def test_unknown_is_not_false_and_gaps_do_not_create_stable_grasp(self):
        rows=[self.row(i,True) for i in (0,2,4)]+[self.row(5,False,-1)]
        rows[-1].pop('robot_grasps')
        summary,_=chronology(rows)
        self.assertNotIn('first_stable_grasp',summary)
        self.assertNotIn('first_grasp_loss_candidate',summary)


if __name__=='__main__':
    unittest.main()
