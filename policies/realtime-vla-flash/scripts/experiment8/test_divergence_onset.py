import numpy as np
import pytest

from scripts.experiment8.summarize_closed_loop import sustained_onset, quaternion_angle_degrees


def test_first_complete_sustained_run():
    assert sustained_onset([0, .03, .03, .03, .03, .03, .04], .02, 5) == 1


@pytest.mark.parametrize('values', [[.03] * 4, [.02] * 5, [.03, .03, .01, .03, .03], []])
def test_incomplete_or_interrupted_run_is_not_onset(values):
    assert sustained_onset(values, .02, 5) is None


@pytest.mark.parametrize('values,threshold,count', [([np.nan], .02, 1), ([[0]], .02, 1), ([0], 0, 1), ([0], .02, 0)])
def test_invalid_inputs_rejected(values, threshold, count):
    with pytest.raises(ValueError):
        sustained_onset(values, threshold, count)


def test_quaternion_sign_and_scale_invariance():
    np.testing.assert_allclose(quaternion_angle_degrees([[1,0,0,0]], [[-2,0,0,0]]), [0])
    np.testing.assert_allclose(quaternion_angle_degrees([[1,0,0,0]], [[0,1,0,0]]), [180])


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quaternion_angle_degrees([[0,0,0,0]], [[1,0,0,0]])
