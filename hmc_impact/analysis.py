"""Threshold, censoring and confound-regression analysis of motion parameters."""

import numpy as np
import pandas as pd

COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
ROTATIONS = COLUMNS[3:]


def _valid(fd_a: np.ndarray, fd_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return both series restricted to frames finite in each."""
    a, b = np.asarray(fd_a, float), np.asarray(fd_b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def _positive(fd_a: np.ndarray, fd_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return both series restricted to frames finite and positive in each."""
    a, b = _valid(fd_a, fd_b)
    keep = (a > 0) & (b > 0)
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


def rotation_term(params: pd.DataFrame, radius: float = 50.0) -> np.ndarray:
    """FD's rotation term: ``radius`` times the summed absolute rotation changes.

    Parameters
    ----------
    params : pandas.DataFrame
        Motion parameters with ``rot_x rot_y rot_z`` in radians.
    radius : float
        Head radius in mm, as in FD.

    Returns
    -------
    (T,) array
        In mm; NaN at index 0.
    """
    steps = np.abs(np.diff(params[ROTATIONS].to_numpy(float), axis=0)).sum(axis=1)
    return np.r_[np.nan, radius * steps]


def relative_parts(
    params_a: pd.DataFrame, params_b: pd.DataFrame, fd_a: np.ndarray, fd_b: np.ndarray
) -> pd.DataFrame:
    """Split the relative FD difference ``(fd_b - fd_a) / fd_a`` into FD's two terms.

    Returns
    -------
    pandas.DataFrame
        Columns ``total``; ``rotation``, the difference of the rotation terms
        (:func:`rotation_term`) over ``fd_a``; and ``translation``, the rest.
        One row per frame; NaN at index 0.
    """
    fd_a, fd_b = np.asarray(fd_a, float), np.asarray(fd_b, float)
    total = (fd_b - fd_a) / fd_a
    rotation = (rotation_term(params_b) - rotation_term(params_a)) / fd_a
    return pd.DataFrame({'total': total, 'rotation': rotation, 'translation': total - rotation})


def relative_spread(fd_a: np.ndarray, fd_b: np.ndarray, q: float = 0.99) -> tuple[float, float]:
    """How much higher or lower one FD series measures than the other, relative to it.

    Both directions count: ``fd_b / fd_a - 1`` and ``fd_a / fd_b - 1``. A frame
    one series measures at or below ``tau`` measures at most ``tau * (1 + up)``
    under the other, for all but a fraction ``1 - q`` of frames; one it measures
    above ``tau`` measures at least ``tau * (1 - down)``.

    Parameters
    ----------
    fd_a, fd_b : numpy.ndarray
        Framewise displacement series; frames with NaN or zero FD in either are skipped.
    q : float
        Quantile; 1 gives the sample maximum.

    Returns
    -------
    tuple of float
        ``(up, down)``: the larger ``q``-quantile, over both directions, of the
        relative excess and of the relative shortfall.
    """
    a, b = _positive(fd_a, fd_b)
    ratios = (b / a, a / b)
    up = max(np.quantile(r - 1, q) for r in ratios)
    down = max(np.quantile(1 - r, q) for r in ratios)
    return float(up), float(down)


def relative_quantiles(
    fd_a: np.ndarray,
    fd_b: np.ndarray,
    edges: np.ndarray,
    q: tuple[float, ...] = (0.01, 0.5, 0.99),
) -> pd.DataFrame:
    """Quantiles of ``fd_b / fd_a - 1`` for frames binned by ``fd_a``.

    Parameters
    ----------
    fd_a, fd_b : numpy.ndarray
        Framewise displacement series; frames with NaN or zero FD in either are skipped.
    edges : array_like
        Bin edges for ``fd_a``; bins include their left edge. Frames outside are dropped.
    q : tuple of float
        Quantiles to report.

    Returns
    -------
    pandas.DataFrame
        Indexed by bin (``pandas.IntervalIndex``), with one column per quantile
        and ``n``, the number of frames; quantiles are NaN in empty bins.
    """
    a, b = _positive(fd_a, fd_b)
    bins = pd.IntervalIndex.from_breaks(edges, closed='left')
    groups = pd.Series(b / a - 1).groupby(pd.cut(a, bins), observed=False)
    table = groups.quantile(list(q)).unstack()
    table['n'] = groups.size()
    return table.set_axis(bins)


def disagreement_curve(
    fd_a: np.ndarray, fd_b: np.ndarray, thresholds, edges: np.ndarray
) -> pd.DataFrame:
    """Disagreement as a function of relative distance from the threshold, pooled over thresholds.

    For each threshold ``tau``, frames are binned by ``|fd_a - tau| / tau``; a
    frame disagrees when ``(fd_a > tau) != (fd_b > tau)``. Each frame counts
    once per threshold whose bins it falls in.

    Parameters
    ----------
    fd_a, fd_b : numpy.ndarray
        Framewise displacement series; frames with NaN in either are skipped.
    thresholds : sequence of float
        Thresholds to pool.
    edges : array_like
        Bin edges of the relative distance; bins include their left edge.

    Returns
    -------
    pandas.DataFrame
        Indexed by bin (``pandas.IntervalIndex``), with ``p``, the fraction that
        disagree (NaN in empty bins), and ``n``, the number of frames.
    """
    a, b = _valid(fd_a, fd_b)
    bins = pd.IntervalIndex.from_breaks(edges, closed='left')
    frames = pd.concat(
        pd.DataFrame({'dist': np.abs(a - tau) / tau, 'disagree': (a > tau) != (b > tau)})
        for tau in thresholds
    )
    groups = frames['disagree'].groupby(pd.cut(frames['dist'], bins), observed=False)
    return pd.DataFrame({'p': groups.mean(), 'n': groups.size()}).set_axis(bins)
