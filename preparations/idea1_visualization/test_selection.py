"""Sampling and label invariants that matter for the scientific protocol."""
import unittest

from .selection import annotate, select_requests


def trace(n):
    return [dict(step=i,executed_action=[0.]*6+[-1.],eef_pos=[.2,0.,0.],
                 robot_contacts={'object':False},robot_grasps={'object':False},
                 eef_object_distances={'object':.2},object_target_distances={},
                 objects={'object':{'pos':[0.,0.,0.]}},primary_objects=['object'],
                 object_displacements={'object':0.},goal_progress=0,success=False) for i in range(n)]


class SamplingTests(unittest.TestCase):
    def test_no_duplicate_requests_and_full_time_span(self):
        rows=annotate(trace(101))
        infers=[dict(request=i,step=i*5,path=str(i)) for i in range(20)]
        samples=select_requests(infers,rows)
        self.assertEqual([s['request'] for s in samples],list(range(20)))
        self.assertIn('uniform',samples[0]['sources'])
        self.assertIn('uniform',samples[-1]['sources'])

    def test_short_rollout_preserves_every_actual_request(self):
        infers=[dict(request=i,step=i*5,path=str(i)) for i in range(3)]
        samples=select_requests(infers,annotate(trace(16)))
        self.assertEqual(len(samples),3)

    def test_event_points_are_before_after_actual_requests(self):
        rows=trace(301)
        for row in rows[132:]:
            row['executed_action'][-1]=1.
            row['robot_contacts']['object']=True
        infers=[dict(request=i,step=i*5,path=str(i)) for i in range(60)]
        samples=select_requests(infers,annotate(rows))
        self.assertEqual(len(samples),20)
        events=[s for s in samples if any(e['kind']=='gripper_switch' for e in s['events'])]
        self.assertEqual({s['step'] for s in events},{130,135})

    def test_close_command_alone_is_not_grasp_or_contact(self):
        rows=trace(10)
        for row in rows:
            row['executed_action'][-1]=1.
            row['robot_contacts']['object']=None
            row['robot_grasps']['object']=None
        labels=annotate(rows)
        self.assertTrue(all(r['phase'] not in ('contact','grasp_or_manipulate','transport') for r in labels))

    def test_missing_evidence_stays_unknown_and_future_contact_does_not_leak(self):
        rows=trace(10)
        for row in rows:
            row['eef_object_distances']={}
            row['robot_contacts']={'object':None}
            row['robot_grasps']={'object':None}
        rows[-1]['robot_contacts']['object']=True
        labels=annotate(rows)
        self.assertEqual([r['phase'] for r in labels[:-1]],['unknown']*9)
        self.assertEqual(labels[-1]['phase'],'contact')

    def test_articulated_region_motion_can_indicate_manipulation_without_contact_truth(self):
        rows=trace(11)
        for i,row in enumerate(rows):
            row['executed_action'][-1]=1.
            row['eef_pos']=[.3+i*.002,0.,0.]
            row['objects']['object']={'type':'site','pos':[i*.002,0.,0.]}
            row['robot_contacts']['object']=None
            row['robot_grasps']['object']=None
            row['goal_predicates']=[dict(predicate=['open','object'],satisfied=False)]
        labels=annotate(rows)
        self.assertEqual(labels[-1]['phase'],'grasp_or_manipulate')
        self.assertIsNone(labels[-1]['phase_evidence']['grasp'])

    def test_completed_subgoal_does_not_make_next_object_aligned(self):
        rows=trace(2)
        for row in rows:
            row['primary_objects']=['object','second']
            row['robot_grasps']={'object':False,'second':True}
            row['robot_contacts']={'object':False,'second':True}
            row['goal_predicates']=[dict(predicate=['in','object','basket'],satisfied=True),
                                    dict(predicate=['in','second','basket'],satisfied=False)]
            row['object_target_distances']={'object->basket':.01,'second->basket':.5}
            row['eef_object_distances']={'object':.3,'second':.03}
        labels=annotate(rows)
        self.assertEqual(labels[-1]['phase'],'grasp_or_manipulate')
        self.assertEqual(labels[-1]['phase_evidence']['active_objects'],['second'])


if __name__=='__main__':
    unittest.main()
