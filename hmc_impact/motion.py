"""Synthetic head-motion model in 3dvolreg ``-1Dfile`` layout.

Columns are ``roll pitch yaw dS dL dP``, in degrees and mm.
"""

import numpy as np

THRESHOLDS: tuple[float, ...] = (0.2, 0.5, 1.0)

#: Standard deviation of increment directions: degrees (roll, pitch, yaw), mm (dS, dL, dP).
_SIGMA = np.array([0.3, 0.6, 0.3, 0.5, 0.3, 0.5])


def volreg_fd(params: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Framewise displacement of 3dvolreg parameters.

    Parameters
    ----------
    params : (T, 6) array
        Motion parameters.
    radius : float
        Head radius in mm.

    Returns
    -------
    (T,) array
        ``sum|d translation| + radius * sum|d deg2rad(rotation)|``; NaN at index 0.
    """
    raise NotImplementedError


def simulate_series_params(
    fd_targets: np.ndarray, rng: np.random.Generator, pull: float = 0.05
) -> np.ndarray:
    """Simulate a motion series whose frame-to-frame FD equals ``fd_targets``.

    Increments are random directions with mean reversion (``-pull * previous pose``),
    scaled to hit each target.

    Returns
    -------
    (len(fd_targets) + 1, 6) array
        Row 0 is zeros; ``volreg_fd(result)[t] == fd_targets[t - 1]``.
    """
    raise NotImplementedError


def simulate_pairs(fd_targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate independent frame pairs ``[pose_i, pose_i + delta_i]``.

    Poses are N(0, 1.5) per column; each increment is scaled to hit ``fd_targets[i]``.

    Returns
    -------
    (2N, 6) array
        ``volreg_fd(result)[2 * i + 1] == fd_targets[i]``.
    """
    raise NotImplementedError


def series_fd_targets(
    n_vols: int,
    rng: np.random.Generator,
    thresholds=THRESHOLDS,
    offsets=(0.03, 0.07, 0.12),
) -> np.ndarray:
    """FD targets for a series: slow drift plus jerks that straddle each threshold.

    Drift frames are U(0.02, 0.1). For each threshold and offset, one jerk lies at
    ``tau * (1 - offset)`` and one at ``tau * (1 + offset)``, at distinct shuffled indices.

    Returns
    -------
    (n_vols - 1,) array
        Index ``t`` is the target for frame ``t + 1``.
    """
    raise NotImplementedError


def sample_fd_targets(
    n: int,
    rng: np.random.Generator,
    thresholds=THRESHOLDS,
    band: float = 0.15,
    fd_max: float = 1.5,
) -> np.ndarray:
    """FD targets for independent pairs, dense near the thresholds.

    Half are U(0, fd_max); half are U(tau * (1 - band), tau * (1 + band)) for a
    threshold chosen uniformly.

    Returns
    -------
    (n,) array
    """
    raise NotImplementedError


def near_threshold_sides(
    fd: np.ndarray, thresholds=THRESHOLDS, band: float = 0.15
) -> dict[float, tuple[bool, bool]]:
    """Whether FD has frames just below and just above each threshold.

    Below means ``tau * (1 - band) <= fd <= tau``; above means ``tau < fd <= tau * (1 + band)``.
    NaN is ignored.

    Returns
    -------
    dict
        ``{tau: (has_below, has_above)}``.
    """
    raise NotImplementedError
