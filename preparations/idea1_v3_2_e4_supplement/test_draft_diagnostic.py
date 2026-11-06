"""CPU checks for scientific selection and fail-closed numerical gates."""
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .draft_diagnostic_protocol import make_plan
from .draft_diagnostic_service import strict_compare,reference_entries,comparisons_accepted
from .draft_diagnostic_inputs import background_geom_allowed,base_robot_qpos
from .draft_diagnostic_language import action_difference
from . import draft_diagnostic_restore as restore
from .draft_diagnostic_audit import DEFAULT_OUT,ROOT,read
from .draft_diagnostic_state import state_difference
from .draft_diagnostic_settle_split import make_plan as make_settle_plan
from .draft_diagnostic_noise_capture import install_capture,clean_counterfactual,Capture,operator_for
from .draft_diagnostic_texture_upload import upload_texture
from .draft_diagnostic_arena_proof import xml_proof,ownership_allowed,EXPECTED


def fake_fog(image,severity=1):
    return np.asarray(image).astype(np.float64)+np.random.uniform(0,severity,size=np.asarray(image).shape)


class DiagnosticContracts(unittest.TestCase):
    def test_exact_unnamed_table_geometry_proof_rejects_shared_objects(self):
        import copy
        proof=xml_proof(EXPECTED)
        self.assertIsNotNone(proof);self.assertEqual(proof['material'],'living_room_table')
        self.assertIsNone(xml_proof(EXPECTED.with_name('libero_living_room_tabletop_base_style.xml')))
        geom=dict(geom_id=9,geom_name=None,body_id=3,body_name='living_room_table_col',static_body=True)
        ownership=[dict(material_name='living_room_table',geoms=[geom])]
        model=SimpleNamespace(_model=object(),body_parentid={3:2},geom_dataid={9:0},geom_type={9:7},
            geom_group={9:1},geom_contype={9:0},geom_conaffinity={9:0},body_id2name=lambda i:'living_room_table')
        env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(model=model),env=SimpleNamespace(_arena_xml=str(EXPECTED))))
        module=SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_MESH=7),mjtObj=SimpleNamespace(mjOBJ_MESH=7),mj_id2name=lambda *args:'living_room_table_vis')
        with patch.dict('sys.modules',mujoco=module):
            self.assertTrue(ownership_allowed(env,'tex-living_room_table',ownership))
            shared=copy.deepcopy(ownership);shared[0]['geoms'].append(dict(geom,body_name='white_mug_1_main'))
            self.assertFalse(ownership_allowed(env,'tex-living_room_table',shared))
            dynamic=copy.deepcopy(ownership);dynamic[0]['geoms'][0]['static_body']=False
            self.assertFalse(ownership_allowed(env,'tex-living_room_table',dynamic))

    def test_texture_upload_passes_native_model_and_exact_context(self):
        from unittest.mock import Mock
        native=object();con=object();wrapped=SimpleNamespace(_model=native,ntex=5)
        gl=SimpleNamespace(make_current=Mock());context=SimpleNamespace(model=wrapped,con=con,gl_ctx=gl)
        sim=SimpleNamespace(model=wrapped,_render_context_offscreen=context)
        module=SimpleNamespace(mjr_uploadTexture=Mock())
        upload_texture(sim,2,module)
        gl.make_current.assert_called_once_with();module.mjr_uploadTexture.assert_called_once_with(native,con,2)
        with self.assertRaises(ValueError):upload_texture(sim,5,module)
        context.model=SimpleNamespace(_model=object())
        with self.assertRaisesRegex(RuntimeError,'does not own'):upload_texture(sim,2,module)

    def test_noise_capture_preserves_original_output_and_rng_stream(self):
        from PIL import Image
        wrapper=SimpleNamespace(fog=fake_fog);env=SimpleNamespace(noise=32)
        image=Image.fromarray(np.arange(12,dtype=np.uint8).reshape(2,2,3))
        np.random.seed(123);expected=fake_fog(image,severity=2);after=np.random.random(8)
        np.random.seed(123)
        with install_capture(env,wrapper) as capture:
            actual=wrapper.fog(image,severity=2)
        continued=np.random.random(8)
        np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(after,continued)
        np.testing.assert_array_equal(capture.clean,np.asarray(image))
        self.assertIs(wrapper.fog,fake_fog);self.assertEqual(capture.metadata()['call_count'],1)

    def test_clean_counterfactual_changes_agentview_only(self):
        capture=Capture(32);capture.clean=np.ones((2,2,3),np.uint8);capture.calls=[{}]*5
        actual=dict(agentview_image=np.zeros((2,2,3),np.uint8),robot0_eye_in_hand_image=np.full((2,2,3),7,np.uint8),
            robot0_eef_pos=np.arange(3,dtype=np.float64),robot0_eef_quat=np.array([0,0,0,1.]),robot0_gripper_qpos=np.array([.03,-.03]))
        env=SimpleNamespace(_diagnostic_noise_capture=capture,_diagnostic_actual_observation=actual)
        counter=clean_counterfactual(env)
        self.assertFalse(np.array_equal(counter['agentview_image'],actual['agentview_image']))
        for key in actual:
            if key!='agentview_image':np.testing.assert_array_equal(counter[key],actual[key])
        self.assertEqual(operator_for(2),('motion_blur',2));self.assertEqual(operator_for(35),('fog',5))

    def test_strict_gate_never_uses_relative_tolerance(self):
        expected=np.full((50,7),1000,np.float32)
        changed=expected.copy();changed[0,0]+=0.001
        self.assertFalse(strict_compare(changed,expected,(50,7))['passed'])
        self.assertTrue(strict_compare(expected,expected,(50,7))['bitwise_equal'])
        self.assertFalse(strict_compare(np.zeros((1,7)),expected,(50,7))['passed'])
        changed[0,0]=np.nan
        self.assertFalse(strict_compare(changed,expected,(50,7))['passed'])

    def test_fixed_40_plan_has_complete_factorial_and_prompt_pairs(self):
        plan=make_plan();self.assertEqual(plan['count'],40)
        grouped={}
        for row in plan['episodes']:
            grouped.setdefault((row['head'],row['case']['task_index']),[]).append(row)
        self.assertEqual(len(grouped),8)
        for rows in grouped.values():
            legacy={(r['init_index'],r['settle']) for r in rows if r['prompt']=='legacy_filename'}
            self.assertEqual(legacy,{(0,'new'),(0,'old'),(6,'new'),(6,'old')})
            prompt=[r for r in rows if r['prompt']=='effective_bddl']
            self.assertEqual([(r['init_index'],r['settle']) for r in prompt],[(0,'new')])
        self.assertEqual(set(Counter(r['head'] for r in plan['episodes']).values()),{10})
        self.assertFalse(plan['video'])

    def test_four_independent_recorded_heads(self):
        entries=reference_entries();self.assertEqual(len(entries),4)
        self.assertEqual(len({r['mode'] for r in entries}),4)
        self.assertEqual([r['source_index'] for r in entries],[0,1,2,3])

    def test_sub_tolerance_numeric_difference_still_blocks_cross_device(self):
        expected=np.zeros((50,7),np.float32);changed=expected.copy();changed[0,0]=1e-8
        comparison=strict_compare(changed,expected,(50,7))
        self.assertTrue(comparison['passed'])
        self.assertFalse(comparisons_accepted(dict(actions=comparison)))
        identical=strict_compare(expected,expected,(50,7))
        self.assertTrue(comparisons_accepted(dict(actions=identical,raw_actions=identical)))

    def test_background_selection_rejects_task_object_name_substrings(self):
        for name in ('table_visual','wall_0','floor','table_leg2_visual'):
            self.assertTrue(background_geom_allowed(name))
        for name in ('tomato_sauce_0_visual','wooden_cabinet_0_table_visual','chocolate_pudding_1','robot0_link7_visual',None):
            self.assertFalse(background_geom_allowed(name))

    def test_robot_target_is_inherited_base_seven_joints(self):
        target=base_robot_qpos();self.assertEqual(target.shape,(7,))
        self.assertTrue(np.isfinite(target).all());self.assertAlmostEqual(target[-1],np.pi/4)

    def test_action_metrics_track_executed_prefix_and_gripper(self):
        first=np.zeros((50,7),np.float32);second=first.copy();second[0,6]=1
        delta=action_difference(first,second)
        self.assertEqual(delta['gripper_first5_sign_disagreements'],1)
        self.assertEqual(delta['first5_max_abs'],1)
        self.assertFalse(delta['bitwise_equal'])
        self.assertEqual(delta['xyz_first5_l2_mean'],0)
        self.assertEqual(delta['rotation_x_first5_rms'],0)

    def test_native_quaternion_branch_can_jump_without_physical_rotation(self):
        q=np.array([np.sin(.05),0,0,np.cos(.05)])
        obs=dict(robot0_eef_quat=q,robot0_eef_pos=np.zeros(3),robot0_gripper_qpos=np.zeros(2))
        other=dict(obs,robot0_eef_quat=-q)
        metrics=state_difference(obs,other)
        self.assertAlmostEqual(metrics['physical_rotation_geodesic_radians'],0)
        self.assertAlmostEqual(metrics['native_axisangle_delta_l2'],2*np.pi)
        self.assertTrue(metrics['near_2pi_branch_jump_candidate'])

    def test_posthoc_settle_plan_only_completes_missing_cells(self):
        plan=make_settle_plan(DEFAULT_OUT/'original_protocol40')
        self.assertTrue(plan['posthoc']);self.assertEqual(plan['count'],4)
        cells={(r['case']['task_index'],r['settle']) for r in plan['episodes']}
        self.assertEqual(cells,{(0,'five_open'),(0,'ten_zero'),(1,'five_open'),(1,'ten_zero')})
        for row in plan['episodes']:
            self.assertEqual((row['head'],row['init_index'],row['prompt']),('draft_object',0,'legacy_filename'))
            self.assertEqual(set(row['reference_episode_indices']),{'old','new'})
        self.assertEqual(plan['settle']['five_open'],dict(steps=5,action=[0.]*6+[-1.]))
        self.assertEqual(plan['settle']['ten_zero'],dict(steps=10,action=[0.]*7))

    def test_restore_reuse_requires_actual_obs_and_bitwise_output(self):
        row=restore.selected(DEFAULT_OUT)[0]
        ledger=read(ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json')
        entry=next(e for e in ledger['selected_completed'] if e['group']=='perturbed350' and e['source_index']==row['source_index'] and e['mode']==row['head'])
        directory=(ROOT/entry['path']).parent;old=read(directory/'result.json')
        with np.load(directory/'initial_native_observation.npz',allow_pickle=False) as archive:
            obs={key:archive[key].copy() for key in restore.OBS_KEYS}
        import json
        with (directory/'infer.jsonl').open() as stream:first=json.loads(next(stream))
        env=SimpleNamespace(obs=obs,info=old['native_info'],case=old['case'])
        response=dict(first['response'],actions=first['actions'])
        with patch.object(restore,'rpc',return_value=response):
            gate,_=restore.gate_reference(env,row,0)
        self.assertTrue(gate['passed'])
        changed=np.asarray(response['raw_actions'],np.float32);changed[0,0]=np.nextafter(changed[0,0],np.float32(np.inf))
        with patch.object(restore,'rpc',return_value=dict(response,raw_actions=changed.tolist())):
            gate,_=restore.gate_reference(env,row,0)
        self.assertFalse(gate['passed'])
        env.obs['robot0_eef_pos'][0]+=0.01
        with patch.object(restore,'rpc') as request:
            gate,_=restore.gate_reference(env,row,0)
        self.assertFalse(gate['passed']);request.assert_not_called()


if __name__=='__main__':unittest.main()
