# hmc-convention-impact

Measures how the AFNI (`3dvolreg`) and MCFLIRT head-motion parameter conventions change
framewise displacement (FD) and the resulting confound regressors.
The analysis lives in the notebook `motion-convention-impact.Rmd` (Jupytext).

## Run the notebook

```bash
uv run jupyter-lab
```

Open `motion-convention-impact.Rmd`. The bundled Jupytext extension registers `.Rmd` as a
notebook; if it opens as text, right-click it and choose **Open With → Jupytext Notebook**.

## Save an executed notebook

```bash
uv run jupytext --to ipynb --execute --run-path . motion-convention-impact.Rmd
```

This writes `motion-convention-impact.ipynb` with all outputs, in under 2 minutes.
`--run-path .` runs the kernel in the repository root, where it finds `hmc_impact` and `data/`.
The `.ipynb` is not committed.

## Regenerate `data/`

Only `3dvolreg` from AFNI is needed:

```bash
PATH="<dir with 3dvolreg>:$PATH" uv run python -m scripts.generate_volreg
```

## Test

```bash
uv run pytest
```

This reports coverage of `hmc_impact` and `tests`.
