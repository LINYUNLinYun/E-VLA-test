"""Future immutable stage matrices and actual vision/NFE acceptance contracts."""
import copy
import json
import unittest
from . import common,vision,worker
from .common import FULL,conditions,validate_manifest,cases


def manifest(stage):
    m=json.loads((common.ROOT/'results/ease_insights_20261103/ae_complement_beam_v4/calibration/manifest.json').read_text())
    m.update(phase='fresh_vision_'+stage+'_v4',conditions=conditions(),noise_repeat=0)
    tasks=[0] if stage=='calibration' else list(range(10));m.update(tasks=tasks,case_table=[dict(suite=s,task=t) for s in common.SUITES for t in tasks])
    if stage=='calibration':m.update(calibration_inits=[45],planned_conditions=['full_d10','dense_d2'],cases_per_condition=4,total_calibration_episodes=8)
    elif stage=='discovery':m.update(discovery_inits=[45,46],planned_conditions=list(common.NEW_CONDITIONS),cases_per_condition=80,total_new_discovery_episodes=160,analytical_cells=320,reference_conditions={'full_d10':'full_d10','dense_d2':'dense_d2'})
    else:m.update(confirmation_inits=[10,11,12,13,14],planned_conditions=list(common.CONDITIONS),cases_per_condition=200,total_confirmation_episodes=800)
    return m


class Contract(unittest.TestCase):
    def test_exact_stage_counts_and_two_disjoint_task_shards(self):
        for stage,per in [('calibration',4),('discovery',80),('confirm',200)]:
            m=manifest(stage);validate_manifest(m);left,right=cases(m,stage,0,2),cases(m,stage,1,2)
            keys=lambda rows:{(r['suite'],r['task'],r['init']) for r in rows}
            self.assertEqual(len(left)+len(right),per);self.assertFalse(keys(left)&keys(right));self.assertEqual(len(keys(left)|keys(right)),per)
    def test_confirmation_keeps_dense_control_and_physical_states_not_noise_repeats(self):
        m=manifest('confirm');self.assertEqual(len(cases(m)),200);self.assertEqual(len(m['planned_conditions']),4)
        m['noise_repeat']=1
        with self.assertRaises(ValueError):validate_manifest(m)
        m=manifest('confirm');m['planned_conditions'].remove('dense_d2')
        with self.assertRaises(ValueError):validate_manifest(m)
    def test_unknown_physical_inits_or_vision_budget_rejected(self):
        m=manifest('confirm');m['confirmation_inits']=[47,48,49]
        with self.assertRaises(ValueError):validate_manifest(m)
        m=manifest('calibration');m['conditions'][2]['vision_kept'][1]=list(range(1,19))
        with self.assertRaises(ValueError):validate_manifest(m)
    def test_source_references_keep_original_ids_and_inert_compensation_is_disclosed(self):
        for alias in ('full_d10','dense_d2'):
            root,m,cm,w,cid=common.reference(alias,verify_sources=False);self.assertEqual(alias,cid)
            old=next(c for c in m['conditions'] if c['id']==cid);self.assertEqual((old['R'],old['H'],old['skip'],old.get('ae_skip',[])),(5,50,[],[]))
    def test_actual_camera_validator_accepts_production_and_rejects_forged_budget(self):
        for cfg in conditions():
            for full in (True,False):
                native=full or cfg['id']=='dense_d2' or cfg['id']=='full_d10'
                if cfg['id']=='full_d10' and not full:continue
                layers=[list(FULL),list(FULL)] if native else cfg['vision_kept']
                audit=dict(kind='native' if native else 'allocated',updated_views=[0,1],cached_views=[],vision_pipeline_invocations=1 if native else 2,
                    actual_kernel_input_batches=[2] if native else [1,1],actual_vision_layers_by_camera=layers,effective_vision_layers_by_camera=layers,
                    image_layer_evaluations=sum(map(len,layers)),fresh_images_consumed=True,vision_cache='none',numerical_reference_only=False,
                    single_view_projection=None if native else 'original_two_view_BF16_partial_order',camera_order=['agentview','wrist'],
                    scope='Observed logical image-layer visits and tensor batches,not GPU kernel-call counts,FLOPs,latency or quality',original_vision_helper_restored=True)
                row=dict(route=dict(camera_execution_audit=audit),timing=dict(camera_execution_audit=audit));worker.validate_camera(row,cfg,full)
                bad=copy.deepcopy(row);bad['route']['camera_execution_audit']['image_layer_evaluations']+=1
                with self.assertRaises(ValueError):worker.validate_camera(bad,cfg,full)
    def test_oracle_output_cannot_be_accepted_as_production(self):
        cfg=conditions()[2]
        row=dict(route=dict(camera_execution_audit=dict(kind='oracle',numerical_reference_only=True)),timing=dict(camera_execution_audit={}))
        with self.assertRaises(ValueError):worker.validate_camera(row,cfg,False)


if __name__=='__main__':unittest.main()
