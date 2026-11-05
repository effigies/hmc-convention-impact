import numpy as np
import pandas as pd

from hmc_impact.analysis import (
    COLUMNS,
    censor_counts,
    disagreement,
    disagreement_curve,
    expand24,
    regress_out,
    relative_parts,
    relative_quantiles,
    relative_spread,
    rotation_term,
    threshold_bounds,
)

FD_A = np.array([np.nan, 0.18, 0.20, 0.22, 0.30, 0.05])
FD_B = np.array([np.nan, 0.21, 0.19, 0.195, 0.31, 0.06])


def test_threshold_bounds():
    assert threshold_bounds(FD_A, FD_B, 0.2) == (0.21, 0.195)


def test_disagreement():
    frac, n = disagreement(FD_A, FD_B, 0.2, band=0.15)
    assert n == 3 and np.isclose(frac, 2 / 3)


def test_censor_counts():
    assert censor_counts(FD_A, FD_B, 0.2) == {'both': 1, 'a_only': 1, 'b_only': 1, 'neither': 2}


def test_expand24():
    params = pd.DataFrame(np.arange(18.0).reshape(3, 6), columns=COLUMNS)  # trans_x … rot_z
    wide = expand24(params)
    assert wide.shape == (3, 24)
    assert list(wide.columns[6:9]) == [
        'trans_x_derivative1',
        'trans_y_derivative1',
        'trans_z_derivative1',
    ]
    assert np.all(wide['trans_x_derivative1'] == [0, 6, 6])
    assert np.all(wide['trans_x_derivative1_power2'] == [0, 36, 36])


def test_regress_out():
    rng = np.random.default_rng(0)
    reg = pd.DataFrame(rng.normal(size=(200, 2)), columns=['u', 'v'])
    target = pd.DataFrame({'lin': 2 * reg.u - reg.v + 1, 'noise': rng.normal(size=200)})
    resid, summary = regress_out(target, reg)
    assert resid.shape == target.shape
    assert np.isclose(summary.loc['lin', 'r2'], 1) and summary.loc['lin', 'rel_resid_sd'] < 1e-10
    assert summary.loc['noise', 'r2'] < 0.05


def _params(rot_x, rot_y):
    """Three frames with the given rotations (rad); translations that FD's rotation term ignores."""
    return pd.DataFrame(
        {
            'trans_x': [0.0, 3.0, -1.0],
            'trans_y': [0.0, 0.5, 0.5],
            'trans_z': [0.0, 0.0, 2.0],
            'rot_x': rot_x,
            'rot_y': rot_y,
            'rot_z': [0.0, 0.0, 0.0],
        }
    )


PARAMS_A = _params([0.0, 0.01, 0.01], [0.0, -0.02, 0.0])  # rotation steps 0.03, 0.02 rad
PARAMS_B = _params([0.0, 0.011, 0.011], [0.0, -0.02, 0.0])  # rotation steps 0.031, 0.02 rad


def test_rotation_term():
    assert np.allclose(rotation_term(PARAMS_A), [np.nan, 1.5, 1.0], equal_nan=True)


def test_relative_parts():
    fd_a, fd_b = np.array([np.nan, 2.0, 1.0]), np.array([np.nan, 2.2, 1.1])
    parts = relative_parts(PARAMS_A, PARAMS_B, fd_a, fd_b)
    assert list(parts.columns) == ['total', 'rotation', 'translation']
    assert parts.iloc[0].isna().all()
    # rotation terms differ by 0.05 mm in frame 1 and not at all in frame 2
    assert np.allclose(parts.iloc[1:], [[0.1, 0.025, 0.075], [0.1, 0.0, 0.1]])


SPREAD_A = np.array([np.nan, 1.0, 1.0, 0.5, 2.0, 0.0])
SPREAD_B = np.array([np.nan, 1.1, 0.8, 0.5, 2.0, 0.1])


def test_relative_spread_max_takes_both_directions():
    # b/a - 1 is 0.1, -0.2, 0, 0; a/b - 1 is -1/11, 0.25, 0, 0; the zero-FD frame is skipped
    up, down = relative_spread(SPREAD_A, SPREAD_B, q=1)
    assert np.isclose(up, 0.25) and np.isclose(down, 0.2)


def test_relative_spread_median():
    assert relative_spread(SPREAD_A, SPREAD_B, q=0.5) == (0.0, 0.0)


def test_relative_spread_bounds_threshold_decisions():
    rng = np.random.default_rng(0)
    fd_a = rng.uniform(0, 1.5, 2000)
    fd_b = fd_a * np.exp(rng.normal(0, 0.05, fd_a.size))
    up, down = relative_spread(fd_a, fd_b, q=1)
    for tau in (0.2, 0.5, 1.0):
        for a, b in ((fd_a, fd_b), (fd_b, fd_a)):
            high, low = threshold_bounds(a, b, tau)
            assert high <= tau * (1 + up) and low >= tau * (1 - down)


def test_relative_quantiles():
    fd_a = np.array([np.nan, 0.1, 0.1, 0.3, 0.3, 0.3, 0.9])
    fd_b = np.array([np.nan, 0.11, 0.09, 0.3, 0.33, 0.36, 0.1])
    table = relative_quantiles(fd_a, fd_b, [0, 0.2, 0.4, 0.6], q=(0, 0.5, 1))
    assert table.index[0].left == 0 and table.index[0].closed == 'left'
    assert list(table['n']) == [2, 3, 0]
    assert np.allclose(table[[0, 0.5, 1]].iloc[:2], [[-0.1, 0.0, 0.1], [0.0, 0.1, 0.2]])
    assert table.iloc[2][[0, 0.5, 1]].isna().all()


def test_disagreement_curve():
    fd_a = np.array([np.nan, 0.199, 0.205, 0.215, 0.98, 1.04, 0.5])
    fd_b = np.array([np.nan, 0.201, 0.204, 0.19, 1.01, 1.03, 0.6])
    edges = [0, 0.05, 0.1, 0.15]
    single = disagreement_curve(fd_a, fd_b, (0.2,), edges)
    assert list(single['n']) == [2, 1, 0]
    assert np.allclose(single['p'], [0.5, 1.0, np.nan], equal_nan=True)
    pooled = disagreement_curve(fd_a, fd_b, (0.2, 1.0), edges)
    assert pooled.index[0].closed == 'left'
    assert list(pooled['n']) == [4, 1, 0]
    assert np.allclose(pooled['p'], [0.5, 1.0, np.nan], equal_nan=True)
