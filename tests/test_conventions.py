import nibabel as nb
import numpy as np
import pytest
from fmriprep.utils.transforms import volreg_to_ras

from hmc_impact.conventions import (
    BOLDREF,
    center_of_gravity,
    framewise_displacement,
    grid_center,
    motion_params,
    pad_image,
    rms_displacement,
    to_volreg_1d,
)
from hmc_impact.motion import simulate_series_params, volreg_fd


@pytest.fixture(scope='module')
def boldref():
    return nb.load(BOLDREF)


@pytest.fixture(scope='module')
def params():
    return simulate_series_params(np.r_[0.3, 1.0, 0.6, 0.2, 0.9], np.random.default_rng(1))


@pytest.fixture(scope='module')
def xfms(params, boldref):
    return volreg_to_ras(params, boldref)


def test_afni_params_reproduce_volreg(tmp_path, boldref, params, xfms):
    df = motion_params(xfms, boldref, 'afni', tmp_path)
    assert list(df.columns) == ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    assert np.allclose(to_volreg_1d(df), params, atol=1e-5)


def test_afni_fd_matches_volreg_fd(tmp_path, boldref, params, xfms):
    fd = framewise_displacement(motion_params(xfms, boldref, 'afni', tmp_path), tmp_path)
    assert np.isnan(fd[0])
    assert np.allclose(fd[1:], volreg_fd(params)[1:], atol=1e-5)


def test_pad_image_moves_only_grid_center(boldref):
    padded = pad_image(boldref, [(10, 0), (0, 0), (0, 0)])
    assert padded.shape == (60, 50, 34)
    assert np.allclose(center_of_gravity(padded), center_of_gravity(boldref))
    assert np.isclose(np.linalg.norm(grid_center(padded) - grid_center(boldref)), 20.0)


def test_fsl_params_invariant_to_padding(tmp_path, boldref, xfms):
    padded = pad_image(boldref, [(10, 0), (0, 5), (0, 0)])
    ref = motion_params(xfms, boldref, 'fsl', tmp_path / 'a')
    pad = motion_params(xfms, padded, 'fsl', tmp_path / 'b')
    assert np.allclose(pad, ref, atol=1e-6)


def test_afni_padding_changes_only_translations(tmp_path, boldref, xfms):
    padded = pad_image(boldref, [(10, 0), (0, 0), (0, 0)])
    ref = motion_params(xfms, boldref, 'afni', tmp_path / 'a')
    pad = motion_params(xfms, padded, 'afni', tmp_path / 'b')
    rot = ['rot_x', 'rot_y', 'rot_z']
    assert np.allclose(pad[rot], ref[rot], atol=1e-8)
    assert not np.allclose(pad.drop(columns=rot), ref.drop(columns=rot), atol=1e-3)


def test_rms_displacement(boldref):
    shift = np.tile(np.eye(4), (3, 1, 1))
    shift[1:, 0, 3] = [1.0, 1.0]
    rms = rms_displacement(shift, boldref)
    assert np.isnan(rms[0])
    assert np.allclose(rms[1:], [1.0, 0.0])
