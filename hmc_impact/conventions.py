"""Motion-parameter conventions, image padding and a convention-free displacement measure."""

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal

import nibabel as nb
import numpy as np
import pandas as pd

BOLDREF: Path = Path(
    str(
        files('fmriprep')
        / 'interfaces/tests/data/sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz'
    )
)


def motion_params(
    xfms: np.ndarray,
    boldref: nb.Nifti1Image,
    convention: Literal['afni', 'fsl'],
    work_dir: Path,
) -> pd.DataFrame:
    """Convert RAS affines to motion parameters with the fMRIPrep/niworkflows interfaces.

    Parameters
    ----------
    xfms : (T, 4, 4) array
        RAS affines in nitransforms' convention.
    boldref : nibabel.Nifti1Image
        Reference image.
    convention : {'afni', 'fsl'}
        Interface to use (``AFNIMotionParams`` or ``FSLMotionParams``).
    work_dir : Path
        Directory for intermediate files; created if needed.

    Returns
    -------
    pandas.DataFrame
        Columns ``trans_x trans_y trans_z rot_x rot_y rot_z`` (mm, rad).
    """
    raise NotImplementedError


def framewise_displacement(params: pd.DataFrame, work_dir: Path) -> np.ndarray:
    """Run niworkflows' ``FramewiseDisplacement`` (radius 50) on motion parameters.

    Returns
    -------
    (T,) array
        FD in mm; NaN at index 0.
    """
    raise NotImplementedError


def to_volreg_1d(params: pd.DataFrame) -> np.ndarray:
    """Convert ``AFNIMotionParams`` output back to 3dvolreg ``-1Dfile`` layout.

    Returns
    -------
    (T, 6) array
        Columns ``roll pitch yaw dS dL dP`` (degrees, mm).
    """
    raise NotImplementedError


def pad_image(img: nb.Nifti1Image, pad: Sequence[tuple[int, int]]) -> nb.Nifti1Image:
    """Pad an image with its minimum value, keeping voxel world coordinates unchanged.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to pad.
    pad : sequence of (before, after)
        Voxels to add along each axis.
    """
    raise NotImplementedError


def grid_center(img: nb.Nifti1Image) -> np.ndarray:
    """RAS coordinates of the center of the voxel grid."""
    raise NotImplementedError


def center_of_gravity(img: nb.Nifti1Image) -> np.ndarray:
    """RAS coordinates of the intensity center of mass (background-subtracted)."""
    raise NotImplementedError


def center_distance(img: nb.Nifti1Image) -> float:
    """Distance in mm between the grid center and the center of gravity."""
    raise NotImplementedError


def rms_displacement(xfms: np.ndarray, img: nb.Nifti1Image) -> np.ndarray:
    """Frame-to-frame RMS displacement over brain voxels, independent of convention.

    Brain voxels are those above 20% of the image's 98th percentile.

    Returns
    -------
    (T,) array
        RMS of ``|xfms[t] @ x - xfms[t-1] @ x|`` in mm; NaN at index 0.
    """
    raise NotImplementedError
