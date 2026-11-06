import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import atomic_json,read_manifest
from .register_shared_e2 import prepare,write
from .dataset_bundle import create,load,GATES
from .same_process_parity import sha


class RegistrationTests(unittest.TestCase):
    def test_truthful_derivation_dryrun_and_no_overwrite(self):
        manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        verified = dict(passed=True,adoption_evidence=[dict(evidence_supports_consideration=True)])
        with tempfile.TemporaryDirectory() as root,patch("preparations.idea1_v3_1.register_shared_e2.verify",return_value=verified):
            root = Path(root)
            atomic_json(root/"audit.json",verified)
            atomic_json(root/"validation.json",dict(arms=[dict(variant=v["id"],wall_s=10+i) for i,v in enumerate(manifest["e2"]["variants"])]))
            targets,registration = prepare(root,manifest)
            self.assertEqual(len(targets),4)
            self.assertFalse(any(p.exists() for p,_ in targets))
            self.assertTrue(all(r["newly_completed_episodes"]==8 and not r["controller_process_receipt"] for _,r in targets))
            write(targets,registration,root)
            first = json.loads(targets[0][0].read_text())
            self.assertEqual(first["source_kind"],"derived_from_authenticated_shared_service_validation")
            with self.assertRaises(FileExistsError):
                prepare(root,manifest)

    def test_changed_audit_or_negative_evidence_rejected(self):
        manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        with tempfile.TemporaryDirectory() as root,patch("preparations.idea1_v3_1.register_shared_e2.verify",return_value=dict(passed=True)):
            atomic_json(Path(root)/"audit.json",dict(passed=False))
            with self.assertRaises(AssertionError):
                prepare(root,manifest)


class BundleTests(unittest.TestCase):
    def fixture(self,root):
        original,shared = Path(root)/"original",Path(root)/"shared"
        original.mkdir()
        shared.mkdir()
        for name in GATES:
            atomic_json(original/name,dict(fixture=name))
        validation,audit = shared/"validation.json",shared/"audit.json"
        atomic_json(validation,{})
        atomic_json(audit,{})
        atomic_json(shared/"shared_registration.json",dict(manifest_hash="fixture",benchmark_aliases=[],derived_timing_targets=[],
            provenance=dict(validation_path=str(validation),validation_sha256=sha(validation),audit_path=str(audit),audit_sha256=sha(audit))))
        manifest = dict(manifest_hash="fixture",conditions=[dict(id="pi0_full",model="pi0"),dict(id="pi05_full",model="pi05")],benchmark_references=[])
        for parent,cid in ((original,"pi0_full"),(shared,"pi05_full")):
            atomic_json(parent/cid/"episodes/suite/task00/init00/result.json",dict(fixture=cid))
        return original,shared,manifest

    def test_split_ownership_actual_paths_and_explicit_glob(self):
        with tempfile.TemporaryDirectory() as root:
            original,shared,manifest = self.fixture(root)
            bundle = Path(root)/"bundle"
            preview = create(bundle,manifest,original,shared)
            self.assertFalse(bundle.exists())
            create(bundle,manifest,original,shared,True)
            index = load(bundle,manifest)
            self.assertEqual(len(list((bundle/"view").glob("*/episodes/*/task*/init*/result.json"))),2)
            self.assertEqual({r["condition"]:r["actual_directory"] for r in index["conditions"]},
                {"pi0_full":str(original/"pi0_full"),"pi05_full":str(shared/"pi05_full")})
            self.assertTrue((bundle/"view/pi05_full").is_symlink())
            with self.assertRaises(FileExistsError):
                create(bundle,manifest,original,shared,True)

    def test_retargeted_condition_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            original,shared,manifest = self.fixture(root)
            bundle = Path(root)/"bundle"
            create(bundle,manifest,original,shared,True)
            link = bundle/"view/pi05_full"
            link.unlink()
            link.symlink_to(original/"pi0_full")
            with self.assertRaises(AssertionError):
                load(bundle,manifest)


if __name__ == "__main__":
    unittest.main()
