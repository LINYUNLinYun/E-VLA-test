"""CPU fake-transport behavior only; never parent authorization or model parity."""
import copy
import json
from pathlib import Path
import time
import unittest
import numpy as np
from preparations.idea1_v3_2.transfer_protocol import digest
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
from .policy_B_proxy import BProxyPolicy,MODES,SCOPE,MIXED,LATENCY

REG=Path('results/idea1_v3_2_e4_supplement/asset_audit/corrected_B_registration_v2/registration.json')


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.reg=json.loads(REG.read_text());reference=self.reg['policy_reference']['manifest_hash']
        self.manifest=dict(kind='independent_corrected_B_CPU_proxy',scope=SCOPE,registration_hash=self.reg['registration_hash'],
            model_reference_manifest_hash=reference,adapter_contract_hash='toy-adapter',
            upstream=dict(manifest_hash=reference,server_pid=123,gpu=3,gpu_uuid='GPU-toy',execution_extension=dict(sha256='toy-extension')))
        self.manifest['manifest_hash']=digest(self.manifest)
        self.auth=dict(scope=SCOPE,allow_quality=True,registration_hash=self.reg['registration_hash'],
            service_manifest_hash=self.manifest['manifest_hash'],model_reference_manifest_hash=reference,
            gpu=3,gpu_uuid='GPU-toy',modes=list(MODES),environments=['calvin','simpler','pusht'],deadline_unix=time.time()+1000,
            allow_shared_upstream_model=True,upstream_service_manifest_hash=reference,upstream_server_pid=123,
            upstream_execution_extension_sha256='toy-extension')
        self.health=dict(protocol=dict(manifest_hash=reference,server_pid=123,gpu_uuid='GPU-toy',family='libero_transfer',
            legacy_gate_passed=True,H=50,R=5,D=10))
        self.calls=[];self.corrupt=None;self.fail=False
        self.policy=BProxyPolicy(self.reg,self.auth,self.manifest,self.forward,self.health)
        self.raw=dict(base_rgb=np.zeros((2,2,3),np.uint8),wrist_rgb=np.zeros((2,2,3),np.uint8),state8=np.zeros(8,np.float32))

    def forward(self,payload):
        self.calls.append(payload)
        if payload['op']=='health':return self.health
        if self.fail:raise ConnectionError('unknown completion, no retry')
        draft=payload['mode'].startswith('draft_')
        observed=dict(vision_encoder_calls=1,vlm_prefix_calls=0 if draft else 1,ae_layer_calls=0 if draft else 180,
            learned_head_calls=1 if draft else 0,verify_calls=0,ae_euler_updates=0 if draft else 10)
        result=dict(manifest_hash=self.manifest['upstream']['manifest_hash'],mode=payload['mode'],actions=np.zeros((50,7)).tolist(),
            raw_actions=np.zeros((50,32)).tolist(),input_hash=input_digest(payload['raw'],payload['instruction']),
            noise_hash=value_digest(np.random.default_rng(payload['seed']).standard_normal((50,32),dtype=np.float32)),
            observed_calls=observed,route=dict(fresh_prefix=True,kv_cross_request=False,fallback=False,verify=False),
            noise_used_in_inference=not draft,category=payload['category'],counters={payload['category']:{payload['mode']:dict(logical_requests=8)}})
        if self.corrupt:result[self.corrupt]='corrupt'
        return result

    def request(self,**changes):
        case=self.reg['cases'][0]
        fields=dict(registration_hash=self.reg['registration_hash'],case_id=case['case_id'],environment=case['environment'],
            mode='full_d10',raw=self.raw,instruction='actual task language',seed=7)
        fields.update(changes);return self.policy.predict(**fields)

    def test_quality_gate_blocks_before_upstream(self):
        with self.assertRaises(RuntimeError):self.request()
        self.assertEqual(self.calls,[])

    def test_explicit_scope_shared_authorization_required(self):
        for key,value in [('allow_shared_upstream_model',False),('upstream_server_pid',999),('scope','Plus')]:
            auth=dict(self.auth,**{key:value})
            with self.assertRaises(PermissionError):BProxyPolicy(self.reg,auth,self.manifest,self.forward,self.health)

    def test_scope_rejected_before_any_forward(self):
        for fields in [dict(registration_hash='Plus'),dict(case_id='unknown'),dict(environment='libero_plus')]:
            with self.assertRaises(PermissionError):self.request(**fields)
        self.assertEqual(self.calls,[])

    def test_quality_not_hidden_as_preflight(self):
        self.policy.bank_gate_passed=True;result=self.request()
        self.assertEqual(self.calls[-1]['category'],'quality')
        self.assertEqual(self.calls[-1]['instruction'],'actual task language')
        np.testing.assert_array_equal(self.calls[-1]['raw']['state8'],self.raw['state8'])
        self.assertEqual(result['category'],'quality')
        self.assertEqual(result['upstream_counter_scope'],MIXED)
        self.assertEqual(result['latency_context'],LATENCY)

    def test_all_modes_actual_route_counts_and_independent_namespace(self):
        for mode in MODES:
            result=self.request(mode=mode,category='preflight')
            self.assertEqual(result['registration_hash'],self.reg['registration_hash'])
            self.assertEqual(result['manifest_hash'],self.manifest['manifest_hash'])
            self.assertEqual(result['upstream_server_pid'],123)
            self.assertEqual(set(result['counters']),{self.policy.namespace})
        self.assertEqual(sum(x['acknowledged_model_requests'] for x in self.policy.counters['preflight'].values()),6)

    def test_uncertain_rpc_is_never_retried(self):
        self.fail=True
        with self.assertRaises(ConnectionError):self.request(category='preflight')
        self.assertEqual(len([p for p in self.calls if p['op']=='predict']),1)
        count=self.policy.counters['preflight']['full_d10']
        self.assertEqual(count['uncertain_model_completion'],1)
        self.assertEqual(count['acknowledged_model_requests'],0)

    def test_bad_input_hash_rejected(self):
        self.corrupt='input_hash'
        with self.assertRaises(ValueError):self.request(category='preflight')

    def test_backend_identity_rechecked_before_each_predict(self):
        self.health['protocol']['server_pid']=999
        with self.assertRaises(PermissionError):self.request(category='preflight')
        self.assertFalse(any(p['op']=='predict' for p in self.calls))

    def test_reset_local_and_bank_requires_separate_permission(self):
        self.assertFalse(self.policy.reset(self.reg['registration_hash'])['upstream_reset_forwarded'])
        with self.assertRaises(PermissionError):self.policy.bank_gate('not-read-without-permission')
        self.assertEqual(self.calls,[])

    def test_health_dual_identity_no_model_constructor(self):
        protocol=self.policy.protocol
        self.assertFalse(protocol['model_constructed']);self.assertFalse(protocol['proxy_cuda_allocation'])
        self.assertEqual(protocol['backend_server_pid'],123)
        self.assertNotEqual(protocol['server_pid'],123)
        self.assertEqual(protocol['authorized_scientific_registrations'],[self.reg['registration_hash']])


if __name__=='__main__':unittest.main()
