"""CPU fixtures only; no model, driver or live scientific records."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit as a


def fixture():
    gpu = next(iter(a.g.HOST_KEEPERS))
    original_host, bulk_host = a.g.HOST_KEEPERS[gpu], 900001
    initial = dict(schema=a.g.km.SCHEMA, phase='ready', sequence=0, gpu_uuid=gpu,
        pid=1234, proc_start_ticks=5678, nonce='a'*32, allocated_bytes=16384*a.g.km.base.MIB,
        total_memory_bytes=24564*a.g.km.base.MIB, driver_version=12040, context_created=True, error=None)
    launch = dict(gpu_uuid=gpu, pid=4567, start_ticks='8765', argv=['python','-m','guardian'])
    amendment = dict(source_hashes={
        'preparations/ease_insights_20261103/reservoir_v1/guardian.py':'a'*64,
        'preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py':'b'*64})
    def observation(size):
        return dict(gpu_uuid=gpu, retained_allocation_mib=size, reservoir_host_pid=bulk_host,
            original_keeper_host_pid=original_host,
            host_process_memory_mib={str(original_host):450, str(bulk_host):size+386})
    binding = dict(observation(16384), keeper=copy.deepcopy(initial), guardian=copy.deepcopy(launch))
    old = dict(gpu_uuid=gpu, allocated_bytes=64*a.g.km.base.MIB)
    original_binding = dict(lease_path='/tmp/fake-original-lease', lease_device=63, lease_inode=123,
                            session=dict(pid=7777))
    revision = dict(schema='ease-reservoir-science-resource-v1', controller_pid=launch['pid'],
        controller_source_sha256='a'*64, reused_borrowed_controller_sha256='b'*64,
        scientific_manifest_hash=a.g.MANIFEST,
        resource_amendment=dict(path=str(a.RESOURCE/'resource_amendment.json'),sha256=a.AMENDMENT_SHA),
        scientific_runtime_unchanged=True, original_holder_resources_unchanged=True,
        original_scheduler_session_owns_lease=True, borrower_did_not_take_or_duplicate_lease=True,
        reservoir_keeper=dict(initial, phase='resized', sequence=1, allocated_bytes=8192*a.g.km.base.MIB),
        assigned_gpu_consistency=observation(8192), keeper=copy.deepcopy(old),
        holder_proof=dict(keeper_ready=copy.deepcopy(old),remote_lease_fds=[4],binding=copy.deepcopy(original_binding)))
    identity = dict(resource_execution_revision=revision, manifest_hash=a.g.MANIFEST,
                    controller_pid=launch['pid'],global_gpu_lock=original_binding['lease_path'])
    result = dict(resource_execution_revision=copy.deepcopy(revision))
    return identity, result, launch, initial, binding, amendment, original_binding


class ResourceReceiptContracts(unittest.TestCase):
    def test_valid_cross_bound_chain_passes(self):
        result = a.validate_revision(*fixture())
        self.assertEqual((result['initial_retained_mib'],result['serving_retained_mib']), (16384,8192))

    def test_a_ready_label_without_real_allocation_fields_fails(self):
        f = fixture()
        f[3]['context_created'] = False
        with self.assertRaises(RuntimeError):
            a.validate_revision(*f)

    def test_different_helper_nonce_or_start_cannot_replace_the_owner(self):
        for key, value in (('nonce','b'*32),('proc_start_ticks',5679)):
            f = fixture();f[0]['resource_execution_revision']['reservoir_keeper'][key]=value
            f[1]['resource_execution_revision']=copy.deepcopy(f[0]['resource_execution_revision'])
            with self.assertRaises(RuntimeError):
                a.validate_revision(*f)

    def test_idle_sized_receipt_cannot_stand_in_for_serving_eight_GiB(self):
        f = fixture();f[0]['resource_execution_revision']['reservoir_keeper']['allocated_bytes']=16384*a.g.km.base.MIB
        f[1]['resource_execution_revision']=copy.deepcopy(f[0]['resource_execution_revision'])
        with self.assertRaises(RuntimeError):
            a.validate_revision(*f)

    def test_changed_controller_source_or_amendment_fails(self):
        for key,value in (('controller_source_sha256','c'*64),('resource_amendment',{})):
            f=fixture();f[0]['resource_execution_revision'][key]=value
            f[1]['resource_execution_revision']=copy.deepcopy(f[0]['resource_execution_revision'])
            with self.assertRaises(RuntimeError):
                a.validate_revision(*f)

    def test_result_with_a_different_resource_revision_fails(self):
        f=fixture();f[1]['resource_execution_revision']['controller_pid']=9999
        with self.assertRaises(RuntimeError):
            a.validate_revision(*f)

    def test_additional_context_or_incorrect_observed_size_fails(self):
        for key,value in (("9999",10),(str(900001),7000)):
            f=fixture();f[0]['resource_execution_revision']['assigned_gpu_consistency']['host_process_memory_mib'][key]=value
            f[1]['resource_execution_revision']=copy.deepcopy(f[0]['resource_execution_revision'])
            with self.assertRaises(RuntimeError):
                a.validate_revision(*f)

    def test_original_lease_cannot_be_replaced_by_another_binding(self):
        f=fixture();f[0]['resource_execution_revision']['holder_proof']['binding']['lease_inode']=999
        f[1]['resource_execution_revision']=copy.deepcopy(f[0]['resource_execution_revision'])
        with self.assertRaises(RuntimeError):
            a.validate_revision(*f)


if __name__ == '__main__':
    unittest.main()
