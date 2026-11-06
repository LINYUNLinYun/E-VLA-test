import numpy as np
import pytest

from .linear import PairedStatistics, mean_squared_error


def test_affine_alignment_recovers_known_map_on_unseen_states():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 7)) + 4
    weight, bias = rng.normal(size=(7, 5)), rng.normal(size=5)
    y = x @ weight + bias
    stats = PairedStatistics(7, 5)
    stats.add(x[:90], y[:90])
    stats.add(x[90:], y[90:])
    unseen = rng.normal(size=(20, 7)) - 2
    np.testing.assert_allclose(stats.fit(1e-8)(unseen), unseen @ weight + bias, atol=1e-5)


def test_fit_does_not_depend_on_stream_chunking():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(80, 6)), rng.normal(size=(80, 6))
    a, b = PairedStatistics(6, 6), PairedStatistics(6, 6)
    a.add(x, y)
    for start in range(0, 80, 8):
        b.add(x[start:start+8], y[start:start+8])
    np.testing.assert_allclose(a.fit(.1).weight, b.fit(.1).weight, atol=1e-7)


def test_mean_scale_control_recovers_global_scale_and_shift():
    x = np.random.default_rng(3).normal(size=(100, 5))
    y = x * 2 + np.arange(5)
    stats = PairedStatistics(5, 5)
    stats.add(x, y)
    assert mean_squared_error(stats.mean_scale(), [(x, y)]) < 1e-12


def test_nonfinite_or_unpaired_states_are_rejected():
    stats = PairedStatistics(4, 4)
    with pytest.raises(ValueError):
        stats.add(np.zeros((2, 4)), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        stats.add(np.full((2, 4), np.nan), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        stats.fit(0)
