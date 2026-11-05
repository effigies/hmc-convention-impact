"""Threshold, censoring and confound-regression analysis of motion parameters."""

import numpy as np
import pandas as pd

COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


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
    raise NotImplementedError


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
    raise NotImplementedError


def censor_counts(fd_a: np.ndarray, fd_b: np.ndarray, tau: float) -> dict[str, int]:
    """Count frames by censoring outcome under each series.

    Returns
    -------
    dict
        Counts under keys ``both``, ``a_only``, ``b_only`` and ``neither``
        (censored by both, only ``a``, only ``b``, neither); NaN frames excluded.
    """
    raise NotImplementedError


def expand24(params: pd.DataFrame) -> pd.DataFrame:
    """Expand six motion parameters to fMRIPrep's 24-parameter model.

    Returns
    -------
    pandas.DataFrame
        The six columns, then ``<col>_derivative1`` (backward difference, 0 in
        the first row), ``<col>_power2`` and ``<col>_derivative1_power2``.
    """
    raise NotImplementedError


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
    raise NotImplementedError
