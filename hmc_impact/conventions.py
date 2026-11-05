"""Motion-parameter conventions, image padding and a convention-free displacement measure."""

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal

import nibabel as nb
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

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
    from fmriprep.interfaces.confounds import AFNIMotionParams
    from fmriprep.utils.transforms import write_itk_affines
    from nipype.pipeline.engine import Node
    from niworkflows.interfaces.confounds import FSLMotionParams

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    xfm_file = work_dir / 'xfms.txt'
    boldref_file = work_dir / 'boldref.nii.gz'
    write_itk_affines(xfms, xfm_file)
    nb.save(boldref, boldref_file)

    iface = {'afni': AFNIMotionParams, 'fsl': FSLMotionParams}[convention](
        xfm_file=str(xfm_file), boldref_file=str(boldref_file)
    )
    out = Node(iface, name=f'{convention}_params', base_dir=str(work_dir)).run().outputs.out_file
    return pd.read_csv(out, sep='\t')


def framewise_displacement(params: pd.DataFrame, work_dir: Path) -> np.ndarray:
    """Run niworkflows' ``FramewiseDisplacement`` (radius 50) on motion parameters.

    Returns
    -------
    (T,) array
        FD in mm; NaN at index 0.
    """
    from nipype.pipeline.engine import Node
    from niworkflows.interfaces.confounds import FramewiseDisplacement

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    in_file = work_dir / 'motion_params.tsv'
    params.to_csv(in_file, sep='\t', index=False)
    iface = FramewiseDisplacement(in_file=str(in_file))
    out = Node(iface, name='fd', base_dir=str(work_dir)).run().outputs.out_file
    return pd.read_csv(out, sep='\t')['FramewiseDisplacement'].to_numpy(dtype=float)


def to_volreg_1d(params: pd.DataFrame) -> np.ndarray:
    """Convert ``AFNIMotionParams`` output back to 3dvolreg ``-1Dfile`` layout.

    Returns
    -------
    (T, 6) array
        Columns ``roll pitch yaw dS dL dP`` (degrees, mm).
    """
    df = params.copy()
    df[['trans_x', 'trans_y', 'rot_x', 'rot_y']] *= -1
    df[['rot_x', 'rot_y', 'rot_z']] = np.rad2deg(df[['rot_x', 'rot_y', 'rot_z']])
    cols = ['rot_z', 'rot_x', 'rot_y', 'trans_z', 'trans_x', 'trans_y']
    return df[cols].to_numpy(dtype=float)


def pad_image(img: nb.Nifti1Image, pad: Sequence[tuple[int, int]]) -> nb.Nifti1Image:
    """Pad an image with its minimum value, keeping voxel world coordinates unchanged.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to pad.
    pad : sequence of (before, after)
        Voxels to add along each axis.
    """
    data = np.asanyarray(img.dataobj)
    padded = np.pad(data, pad, constant_values=data.min())
    affine = img.affine.copy()
    affine[:3, 3] -= img.affine[:3, :3] @ np.array([before for before, _ in pad])
    return img.__class__(padded, affine, img.header)


def grid_center(img: nb.Nifti1Image) -> np.ndarray:
    """RAS coordinates of the center of the voxel grid."""
    return nb.affines.apply_affine(img.affine, (np.array(img.shape[:3]) - 1) / 2)


def center_of_gravity(img: nb.Nifti1Image) -> np.ndarray:
    """RAS coordinates of the intensity center of mass (background-subtracted)."""
    data = np.asanyarray(img.dataobj, dtype=float)
    return nb.affines.apply_affine(img.affine, center_of_mass(data - data.min()))


def center_distance(img: nb.Nifti1Image) -> float:
    """Distance in mm between the grid center and the center of gravity."""
    return float(np.linalg.norm(grid_center(img) - center_of_gravity(img)))


def rms_displacement(xfms: np.ndarray, img: nb.Nifti1Image) -> np.ndarray:
    """Frame-to-frame RMS displacement over brain voxels, independent of convention.

    Brain voxels are those above 20% of the image's 98th percentile.

    Returns
    -------
    (T,) array
        RMS of ``|xfms[t] @ x - xfms[t-1] @ x|`` in mm; NaN at index 0.
    """
    data = np.asanyarray(img.dataobj, dtype=float)
    ijk = np.argwhere(data > 0.2 * np.percentile(data, 98))
    xyz = np.c_[nb.affines.apply_affine(img.affine, ijk), np.ones(len(ijk))]
    pos = np.einsum('tij,nj->tni', np.asarray(xfms), xyz)[..., :3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    return np.r_[np.nan, np.sqrt((step**2).mean(axis=1))]
