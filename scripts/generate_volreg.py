"""
Generate the 3dvolreg data that the notebook starts from.

3dvolreg is too slow to run in the notebook, so this script runs it once. It
draws a series of frame-to-frame FD targets that straddle each threshold, turns
them into 3dvolreg-convention motion parameters, moves the test boldref by those
motions to make a synthetic series, and registers the series back to the boldref
with the options fMRIPrep's HMC workflow uses. The boldref, the ``-1Dfile`` output
(``volreg.1D``) and the ``-1Dmatrix_save`` output (``volreg.aff12.1D``) are written
to ``data/``. The series is not kept.

The script checks that 3dvolreg's estimates have frames just below and just above
every threshold, and raises ``ValueError`` if not; pass another ``--seed`` to draw
different motions.

``3dvolreg`` must be on the ``PATH``. The committed data were made with AFNI_26.3.09.

Run as ``uv run python -m scripts.generate_volreg``.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nb
import nitransforms as nt
import nitransforms.resampling
import numpy as np
import typer
from fmriprep.interfaces.patches import Volreg
from fmriprep.utils.transforms import volreg_to_ras
from nipype.pipeline import engine as pe

from hmc_impact.conventions import BOLDREF
from hmc_impact.motion import (
    THRESHOLDS,
    near_threshold_sides,
    series_fd_targets,
    simulate_series_params,
    volreg_fd,
)


def synthesize_series(params: np.ndarray, boldref: nb.Nifti1Image) -> nb.Nifti1Image:
    """Move ``boldref`` by the 3dvolreg-convention ``params`` (one row per volume).

    Volume ``t`` is ``boldref`` resampled so that 3dvolreg registers it back with
    ``params[t]``; the result is float32.
    """
    vols = []
    for H in volreg_to_ras(params, boldref):
        xfm = nt.linear.Affine(np.linalg.inv(H), reference=boldref)
        vols.append(nt.resampling.apply(xfm, boldref, order=3))
    series = nb.funcs.concat_images(vols)
    series.set_data_dtype('float32')
    return series


def check_near_threshold(oned: np.ndarray, band: float = 0.15) -> None:
    """Raise ``ValueError`` unless FD from ``oned`` has frames just below and above every threshold."""
    sides = near_threshold_sides(volreg_fd(oned), band=band)
    for tau, (below, above) in sides.items():
        for side, hit in zip(('below', 'above'), (below, above)):
            if not hit:
                raise ValueError(
                    f'No estimated FD frame lies just {side} {tau:g} mm '
                    f'(within {band:.0%}); try another --seed.'
                )


def main(
    out_dir: Path = Path('data'),
    seed: int = 0,
    n_vols: int = 200,
) -> None:
    """Run 3dvolreg on a synthetic series and write its outputs to ``out_dir``."""
    if shutil.which('3dvolreg') is None:
        raise typer.BadParameter('3dvolreg was not found on PATH', param_hint='PATH')

    boldref = nb.load(BOLDREF)
    rng = np.random.default_rng(seed)
    targets = series_fd_targets(n_vols, rng, thresholds=THRESHOLDS)
    params = simulate_series_params(targets, rng)
    series = synthesize_series(params, boldref)

    with TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        series.to_filename(work / 'series.nii.gz')
        boldref.to_filename(work / 'boldref.nii.gz')
        volreg = pe.Node(
            Volreg(
                in_file=str(work / 'series.nii.gz'),
                basefile=str(work / 'boldref.nii.gz'),
                interp='cubic',
                zpad=1,
                out_file='NULL',
            ),
            name='volreg',
            base_dir=str(work),
        )
        outputs = volreg.run().outputs
        check_near_threshold(np.loadtxt(outputs.oned_file))

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(work / 'boldref.nii.gz', out_dir / 'boldref.nii.gz')
        shutil.copy(outputs.oned_file, out_dir / 'volreg.1D')
        shutil.copy(outputs.oned_matrix_save, out_dir / 'volreg.aff12.1D')


if __name__ == '__main__':
    typer.run(main)
