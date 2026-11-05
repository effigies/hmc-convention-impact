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
    params = np.asarray(params, dtype=float)
    steps = np.abs(np.diff(params, axis=0))
    fd = steps[:, 3:].sum(axis=1) + radius * np.deg2rad(steps[:, :3]).sum(axis=1)
    return np.r_[np.nan, fd]


def _scale_to_fd(direction: np.ndarray, target: float) -> np.ndarray:
    """Scale an increment so its FD equals ``target`` (FD is linear in the increment)."""
    fd = volreg_fd(np.vstack([np.zeros(6), direction]))[1]
    return direction * (target / fd)


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
    fd_targets = np.asarray(fd_targets, dtype=float)
    params = np.zeros((len(fd_targets) + 1, 6))
    for t, target in enumerate(fd_targets):
        direction = rng.normal(0, _SIGMA) - pull * params[t]
        params[t + 1] = params[t] + _scale_to_fd(direction, target)
    return params


def simulate_pairs(fd_targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate independent frame pairs ``[pose_i, pose_i + delta_i]``.

    Poses are N(0, 1.5) per column; each increment is scaled to hit ``fd_targets[i]``.

    Returns
    -------
    (2N, 6) array
        ``volreg_fd(result)[2 * i + 1] == fd_targets[i]``.
    """
    fd_targets = np.asarray(fd_targets, dtype=float)
    params = np.zeros((2 * len(fd_targets), 6))
    for i, target in enumerate(fd_targets):
        params[2 * i] = rng.normal(0, 1.5, 6)
        params[2 * i + 1] = params[2 * i] + _scale_to_fd(rng.normal(0, _SIGMA), target)
    return params


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
    n_steps = n_vols - 1
    targets = rng.uniform(0.02, 0.1, n_steps)
    jerks = np.array(
        [tau * (1 + sign * offset) for tau in thresholds for offset in offsets for sign in (-1, 1)]
    )
    targets[rng.choice(n_steps, size=len(jerks), replace=False)] = rng.permutation(jerks)
    return targets


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
    thresholds = np.asarray(thresholds, dtype=float)
    n_near = n // 2
    taus = rng.choice(thresholds, size=n_near)
    near = rng.uniform(taus * (1 - band), taus * (1 + band))
    broad = rng.uniform(0, fd_max, n - n_near)
    return rng.permutation(np.r_[broad, near])


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
    fd = np.asarray(fd, dtype=float)
    fd = fd[~np.isnan(fd)]
    return {
        tau: (
            bool(np.any((fd >= tau * (1 - band)) & (fd <= tau))),
            bool(np.any((fd > tau) & (fd <= tau * (1 + band)))),
        )
        for tau in thresholds
    }
