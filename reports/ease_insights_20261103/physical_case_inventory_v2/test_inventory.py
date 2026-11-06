"""Prevent treating flat FLASH cases as unused initial states."""
import unittest
from .inventory import extract


class MetadataSchemas(unittest.TestCase):
    def test_canonical_case(self):
        self.assertEqual(extract(dict(case=dict(suite='libero_spatial',task=8,init=49)),'/x'),('libero_spatial',8,49))
    def test_flash_flat(self):
        self.assertEqual(extract(dict(suite='libero_spatial',task_id=8,init_id=21),'/x'),('libero_spatial',8,21))
    def test_native(self):
        self.assertEqual(extract(dict(task_suite_name='libero_10',task_id=1,episode_idx=22),'/x'),('libero_10',1,22))
    def test_first_observation_flat_path_without_reading_array(self):
        self.assertEqual(extract(None,'/x/episodes/libero_spatial_t08_i24/first_policy_observation.npz'),('libero_spatial',8,24))
    def test_canonical_path_without_reading_array(self):
        self.assertEqual(extract(None,'/x/episodes/libero_10/task01/init25/input_0000.npz'),('libero_10',1,25))
    def test_metadata_and_path_conflict_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):extract(dict(case=dict(suite='libero_spatial',task=8,init=47)),'/x/episodes/libero_spatial_t08_i48/result.json')


if __name__=='__main__':unittest.main()
