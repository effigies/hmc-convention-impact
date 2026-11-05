# hmc-convention-impact

Measures how the AFNI (`3dvolreg`) and MCFLIRT head-motion parameter conventions change
framewise displacement (FD) and the resulting confound regressors.
The analysis lives in the notebook `motion-convention-impact.Rmd` (Jupytext).

## Run the notebook

```bash
uv run jupyter-lab
```

Open `motion-convention-impact.Rmd`.

## Regenerate `data/`

Only `3dvolreg` from AFNI is needed:

```bash
PATH="<dir with 3dvolreg>:$PATH" uv run python -m scripts.generate_volreg
```

## Test

```bash
uv run pytest
```
