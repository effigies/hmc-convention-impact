"""Generate the 3dvolreg data that the notebook starts from."""

from pathlib import Path

import nibabel as nb
import numpy as np


def synthesize_series(params: np.ndarray, boldref: nb.Nifti1Image) -> nb.Nifti1Image:
    """Move ``boldref`` by the 3dvolreg-convention ``params`` (one row per volume)."""
    raise NotImplementedError


def check_near_threshold(oned: np.ndarray, band: float = 0.15) -> None:
    """Raise ``ValueError`` unless FD from ``oned`` falls just below and above every threshold."""
    raise NotImplementedError


def main(out_dir: Path = Path('data'), seed: int = 0, n_vols: int = 200) -> None:
    """Run 3dvolreg on a synthetic series and write its outputs to ``out_dir``."""
    raise NotImplementedError
