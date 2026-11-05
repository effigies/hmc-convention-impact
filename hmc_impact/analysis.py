"""Threshold, censoring and confound-regression analysis of motion parameters."""

import numpy as np
import pandas as pd

COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


def _valid(fd_a: np.ndarray, fd_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return both series restricted to frames finite in each."""
    a, b = np.asarray(fd_a, float), np.asarray(fd_b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def threshold_bounds(fd_a: np.ndarray, fd_b: np.ndarray, tau: float) -> tuple[float, float]:
    """Bound ``fd_b`` by the censoring decision made on ``fd_a``.

    Parameters
    ----------
    fd_a, fd_b : numpy.ndarray
        Framewise displacement series; frames with NaN in either are skipped.
    tau : float
        Censoring threshold.

    Returns
    -------
    tuple of float
        ``(max fd_b where fd_a <= tau, min fd_b where fd_a > tau)``; NaN for a
        side with no frames. Swap the arguments for ``b``'s point of view.
    """
    a, b = _valid(fd_a, fd_b)
    low, high = b[a <= tau], b[a > tau]
    return (
        float(low.max()) if low.size else float('nan'),
        float(high.min()) if high.size else float('nan'),
    )


def disagreement(
    fd_a: np.ndarray, fd_b: np.ndarray, tau: float, band: float = 0.1
) -> tuple[float, int]:
    """Fraction of near-threshold frames on which censoring decisions differ.

    Parameters
    ----------
    fd_a, fd_b : numpy.ndarray
        Framewise displacement series; frames with NaN in either are skipped.
    tau : float
        Censoring threshold.
    band : float
        Relative half-width: frames with ``|fd_a - tau| <= band * tau`` are used.

    Returns
    -------
    tuple
        Fraction of those frames where ``(fd_a > tau) != (fd_b > tau)`` and the
        number of frames; ``(nan, 0)`` if there are none.
    """
    a, b = _valid(fd_a, fd_b)
    near = np.abs(a - tau) <= band * tau
    n = int(near.sum())
    if n == 0:
        return float('nan'), 0
    return float(np.mean((a[near] > tau) != (b[near] > tau))), n


def censor_counts(fd_a: np.ndarray, fd_b: np.ndarray, tau: float) -> dict[str, int]:
    """Count frames by censoring outcome under each series.

    Returns
    -------
    dict
        Counts under keys ``both``, ``a_only``, ``b_only`` and ``neither``
        (censored by both, only ``a``, only ``b``, neither); NaN frames excluded.
    """
    a, b = _valid(fd_a, fd_b)
    a_cens, b_cens = a > tau, b > tau
    return {
        'both': int(np.sum(a_cens & b_cens)),
        'a_only': int(np.sum(a_cens & ~b_cens)),
        'b_only': int(np.sum(~a_cens & b_cens)),
        'neither': int(np.sum(~a_cens & ~b_cens)),
    }


def expand24(params: pd.DataFrame) -> pd.DataFrame:
    """Expand six motion parameters to fMRIPrep's 24-parameter model.

    Returns
    -------
    pandas.DataFrame
        The six columns, then ``<col>_derivative1`` (backward difference, 0 in
        the first row), ``<col>_power2`` and ``<col>_derivative1_power2``.
    """
    base = params[COLUMNS]
    deriv = base.diff().fillna(0.0).add_suffix('_derivative1')
    return pd.concat(
        [
            base,
            deriv,
            base.pow(2).add_suffix('_power2'),
            deriv.pow(2).add_suffix('_power2'),
        ],
        axis=1,
    )


def regress_out(
    target: pd.DataFrame, regressors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress ``regressors`` (plus an intercept) out of each ``target`` column.

    Returns
    -------
    resid : pandas.DataFrame
        Residuals, same shape as ``target``.
    summary : pandas.DataFrame
        Indexed by target column, with ``r2`` and ``rel_resid_sd`` (residual SD
        over target SD).
    """
    design = np.column_stack([regressors.to_numpy(float), np.ones(len(regressors))])
    y = target.to_numpy(float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    summary = pd.DataFrame(
        {'r2': 1 - ss_res / ss_tot, 'rel_resid_sd': resid.std(axis=0) / y.std(axis=0)},
        index=target.columns,
    )
    return pd.DataFrame(resid, index=target.index, columns=target.columns), summary
