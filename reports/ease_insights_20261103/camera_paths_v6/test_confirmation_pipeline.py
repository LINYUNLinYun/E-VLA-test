"""Synthetic800-row JSON interface tests;no real outcomes/model/environment used."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from preparations.ease_insights_20261103.camera_paths_v6.common import SUITES,CONDITIONS,NEW_CONDITIONS,digest
from reports.ease_insights_20261103.camera_paths_v6 import confirmation_analysis as production
from reports.ease_insights_20261103.camera_paths_v6 import confirmation_statistics_audit as independent


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class PipelineTest(unittest.TestCase):
    def exercise(self,kind):
        with tempfile.TemporaryDirectory(prefix='ease_synthetic800_only_') as tmp:
            root=Path(tmp);study=root/'study';report=root/'report';paths=list(NEW_CONDITIONS);inits=[10,11,12,13,14]
            directions=[dict(suite='libero_goal',better=paths[0],worse=paths[1],discovery_difference=.2),dict(suite='libero_10',better=paths[1],worse=paths[0],discovery_difference=.2)]
            decision=dict(status='selected',selected=dict(paths=paths,directions=directions))
            selected=dict(revision='ease-full-storage-image-mask-selection-v6',complete_discovery_cells=320,new_discovery_episodes=160,preserved_reference_episodes=160,
                independent_validation_launched=False,selection=decision,evidence={},discovery_manifest_hash='synthetic_discovery')
            selected['selection_hash']=digest(selected);selection=root/'selection.json';write(selection,selected)
            m=dict(synthetic=True,suites=list(SUITES),tasks=list(range(10)),planned_conditions=list(CONDITIONS),analytical_conditions=list(CONDITIONS),
                confirmation_inits=inits,noise_repeat=0,total_confirmation_episodes=800,source_files={},preregistered_hypotheses=directions,
                protocol_evidence=dict(path='synthetic_protocol',sha256='synthetic'),selection_evidence=dict(path=str(selection),sha256=sha(selection),
                    selection_hash=selected['selection_hash'],frozen_selection=decision))
            m['manifest_hash']=digest(m);write(study/'manifest.json',m);mh=m['manifest_hash']
            scenes=[dict(suite=s,task=t,init=i) for s in SUITES for t in range(10) for i in inits];records={};case_pins=[]
            for cid in CONDITIONS:
                for case in scenes:
                    s,t,i=case['suite'],case['task'],case['init'];success=True
                    if kind!='zero':
                        success=not ((cid==paths[1] and s=='libero_goal') or (cid==paths[0] and s=='libero_10')) or not (t<6 and i==10)
                    if kind=='quality_negative' and cid==paths[0] and (s=='libero_object' or (s=='libero_spatial' and t<3)):success=False
                    folder=study/cid/'episodes'/s/f'task{t:02d}'/f'init{i:02d}'
                    result=dict(synthetic=True,success=success,condition=cid,case=case,manifest_hash=mh)
                    write(folder/'result.json',result)
                    receipt=dict(synthetic=True,valid=True,success=success,condition=cid,case=case,manifest_hash=mh,artifacts={'result.json':sha(folder/'result.json')})
                    path=folder/'episode_receipt.json';write(path,receipt);case_pins.append(dict(receipt=str(path),sha256=sha(path)))
                    records[cid,(s,t,i)]=dict(success=success,condition=cid,case=case,actions=15,predictions=3,full_requests=1,fast_requests=2,
                        policy_ms=[150.,100.,100.],full_policy_ms=[150.],fast_policy_ms=[100.,100.],rpc_ms=[151.,101.,101.],episode_wall_s=.5,
                        full_executed_actions=5,fast_executed_actions=10,gpu_uuid='synthetic_GPU',initial_gate={})
            services=[]
            for index in range(8):
                cid=CONDITIONS[index//2];run=study/cid/'runs'/str(index);blocks=[]
                for suite in SUITES:
                    blocks.append(dict(suite=suite,conditions=[dict(condition=c,measured=[dict(policy_wall_ms=150. if c=='full_d10' else 100.) for _ in range(12)]) for c in CONDITIONS]))
                write(run/'fixed_input_cost.json',dict(synthetic=True,manifest_hash=mh,suites=blocks))
                write(run/'result.json',dict(synthetic=True,manifest_hash=mh,complete=True,condition=cid))
                services.append(dict(run=str(run),artifacts={'fixed_input_cost.json':sha(run/'fixed_input_cost.json')}))
            accepted=dict(synthetic=True,status='pass',manifest_hash=mh,new_cases=800,preserved_control_cells=0,new_case_evidence=case_pins,service_evidence=services)
            write(report/'case_audit.json',accepted);write(report/'prelaunch_confirmation.json',dict(synthetic=True,manifest_hash=mh,manifest_file_sha256=sha(study/'manifest.json')))
            audit=dict(root=study,manifest=m,records=records,scenes=scenes,complete=True,missing=[],invalid=[],unexpected=[],episode_evidence=[])
            costs=dict(valid_services=8,services=[],excluded_or_pending=[],median_full_over_policy_cycle={c:1. if c=='full_d10' else 1.5 for c in CONDITIONS})
            with patch.object(production,'load_stage',return_value=audit),patch.object(production,'completed_service_costs',return_value=costs):
                value=production.analyze_confirmation(study,contention_root=root/'empty_contention')
            write(report/'analysis.json',value)
            with patch.object(independent,'STUDY',study),patch.object(independent,'REPORT',report):
                result=independent.audit(report/'analysis.json',report/'case_audit.json',report/'independent.json')
            self.assertEqual(result['status'],'pass',result['differences']);self.assertEqual(result['differences'],[])
            self.assertEqual(value['valid_episodes'],800);self.assertFalse(value['adaptive_routing_benefit_established'])
            return value['conclusion'],value['DenseD2_dominates_both_at_no_higher_native_cost']
    def test_positive_complete_json_matrix(self):
        conclusion,dominated=self.exercise('positive');self.assertEqual(conclusion,'complementarity_reproduced');self.assertTrue(dominated)
    def test_zero_contrasts_are_not_reproduced(self):self.assertEqual(self.exercise('zero')[0],'not_reproduced')
    def test_quality_failure_overrides_positive_primary_statistics(self):self.assertEqual(self.exercise('quality_negative')[0],'not_reproduced')


if __name__=='__main__':unittest.main()
