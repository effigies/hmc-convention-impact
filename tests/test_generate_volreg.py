import shutil

import nibabel as nb
import numpy as np
import pytest

from hmc_impact.conventions import BOLDREF
from hmc_impact.motion import simulate_series_params
from scripts.generate_volreg import check_near_threshold, main, synthesize_series


def test_synthesize_series_volume0_is_boldref():
    boldref = nb.load(BOLDREF)
    params = simulate_series_params(np.r_[0.5], np.random.default_rng(0))
    series = synthesize_series(params, boldref)
    assert series.shape == (*boldref.shape, 2)
    assert np.allclose(series.dataobj[..., 0], boldref.dataobj, atol=1e-3)
    assert not np.allclose(series.dataobj[..., 1], boldref.dataobj, atol=1e-3)


def test_check_near_threshold_raises():
    oned = simulate_series_params(np.full(20, 0.05), np.random.default_rng(0))
    with pytest.raises(ValueError, match='0.2'):
        check_near_threshold(oned)


@pytest.mark.skipif(shutil.which('3dvolreg') is None, reason='3dvolreg required')
def test_main_writes_data(tmp_path):
    main(out_dir=tmp_path)  # the default seed, which Step 5 confirms straddles every threshold
    assert {p.name for p in tmp_path.iterdir()} == {
        'boldref.nii.gz',
        'volreg.1D',
        'volreg.aff12.1D',
    }
    assert np.loadtxt(tmp_path / 'volreg.1D').shape == (200, 6)
