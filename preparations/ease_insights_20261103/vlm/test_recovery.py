import json
from pathlib import Path
import tempfile
import unittest
from .recovery import CompletedEpisodeFinalizationError, run_with_retries, scheduled_nfe


class RecoveryTests(unittest.TestCase):
    def test_real_exported_node_lists(self):
        d10=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.30000000000000004,0.19999999999999996,0.09999999999999998,0.0]
        self.assertEqual(scheduled_nfe(d10),10)
        self.assertEqual(scheduled_nfe([1.0,0.5,0.0]),2)
        with self.assertRaises(ValueError): scheduled_nfe([1,.4,0])
        with self.assertRaises(ValueError): scheduled_nfe({"actual_NFE":10})

    def test_finalize_failure_never_retries_complete_science(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/"init30"; calls=[]
            def episode():
                calls.append("science");directory.mkdir()
                (directory/"complete.json").write_text(json.dumps(dict(valid=True)))
            def fail(): raise TypeError("receipt error")
            for _ in range(2):
                with self.assertRaises(CompletedEpisodeFinalizationError):
                    run_with_retries(directory,3,episode,fail,lambda:None)
            self.assertEqual(calls,["science"])
            self.assertTrue((directory/"complete.json").exists())
            self.assertEqual(list(Path(root).glob("init30.invalid.*")),[])

    def test_post_complete_cleanup_failure_never_retries(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/"init30";calls=[]
            def episode():
                calls.append("science");directory.mkdir()
                (directory/"complete.json").write_text("{}")
                raise RuntimeError("environment cleanup failed after completion")
            with self.assertRaises(CompletedEpisodeFinalizationError):
                run_with_retries(directory,3,episode,lambda:{},lambda:None)
            self.assertEqual(calls,["science"])
            self.assertTrue((directory/"complete.json").exists())


if __name__ == "__main__": unittest.main()
