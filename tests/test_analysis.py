import numpy as np
import pandas as pd

from hmc_impact.analysis import (
    COLUMNS,
    censor_counts,
    disagreement,
    expand24,
    regress_out,
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
