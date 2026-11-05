import numpy as np

from hmc_impact.motion import (
    THRESHOLDS,
    near_threshold_sides,
    sample_fd_targets,
    series_fd_targets,
    simulate_pairs,
    simulate_series_params,
    volreg_fd,
)


def test_volreg_fd():
    params = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0.5, 0, 0], [1, 0, 0, 0.5, 0, 0.0]])
    fd = volreg_fd(params)
    assert np.isnan(fd[0])
    assert np.allclose(fd[1:], [0.5 + 50 * np.deg2rad(1), 0.0])


def test_simulate_series_hits_targets():
    rng = np.random.default_rng(0)
    targets = rng.uniform(0.02, 1.2, 50)
    params = simulate_series_params(targets, rng)
    assert params.shape == (51, 6)
    assert np.all(params[0] == 0)
    assert np.allclose(volreg_fd(params)[1:], targets)


def test_simulate_series_stays_bounded():
    rng = np.random.default_rng(0)
    params = simulate_series_params(np.full(2000, 0.1), rng)
    assert np.abs(params[:, :3]).max() < 5  # degrees
    assert np.abs(params[:, 3:]).max() < 5  # mm


def test_simulate_pairs_hits_targets():
    rng = np.random.default_rng(0)
    targets = rng.uniform(0, 1.5, 100)
    params = simulate_pairs(targets, rng)
    assert params.shape == (200, 6)
    assert np.allclose(volreg_fd(params)[1::2], targets)


def test_series_fd_targets_straddle_thresholds():
    targets = series_fd_targets(200, np.random.default_rng(0))
    assert targets.shape == (199,)
    assert near_threshold_sides(np.r_[np.nan, targets]) == {t: (True, True) for t in THRESHOLDS}
    assert np.sum(targets > 0.85 * 0.2) == 18  # only the jerks reach the lowest band


def test_sample_fd_targets_dense_near_thresholds():
    targets = sample_fd_targets(10_000, np.random.default_rng(0))
    assert targets.min() >= 0 and targets.max() <= 1.5
    for tau in THRESHOLDS:
        assert np.mean(np.abs(targets - tau) <= 0.15 * tau) > 0.1


def test_near_threshold_sides():
    fd = np.array([np.nan, 0.19, 0.21, 0.5])
    assert near_threshold_sides(fd, (0.2, 0.5)) == {0.2: (True, True), 0.5: (True, False)}
