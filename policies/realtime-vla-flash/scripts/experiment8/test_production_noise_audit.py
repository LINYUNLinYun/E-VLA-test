import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'spec'))
import spec_serve_policy


def test_production_noise_hash_matches_generated_input_without_counter_change():
    cls = next(value for value in vars(spec_serve_policy).values()
               if isinstance(value, type) and '_make_ablation_noise' in vars(value))
    policy = object.__new__(cls)
    policy._ablation_noise_seed = 12345
    policy._ablation_noise_fallback_counter = 7
    policy._action_horizon, policy._action_dim = 50, 32
    policy._ablation_fields, policy._experiment_fields = {}, {}
    noise, mode = policy._make_ablation_noise(episode_idx=90, infer_id=3)
    output = {}
    policy._attach_ablation_metadata(output, episode_idx=90, infer_id=3, noise_seed_mode=mode)
    assert output['policy_timing']['noise_sha256'] == hashlib.sha256(noise.tobytes()).hexdigest()
    assert policy._ablation_noise_fallback_counter == 7
    repeated, _ = policy._make_ablation_noise(episode_idx=90, infer_id=3)
    np.testing.assert_array_equal(noise, repeated)
    provided = {}
    policy._attach_ablation_metadata(provided, episode_idx=90, infer_id=3, noise_seed_mode='provided')
    assert 'noise_sha256' not in provided['policy_timing']
