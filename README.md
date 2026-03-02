# float16-sfogliatella

Sfogliatella is a general-purpose time-series machine learning library with Python API and CLI workflows for training, prediction, live inference, HPO, stacking, and model packaging.

## Install

### PyPI install (after release publish)

```bash
pip install float16-sfogliatella
```

`pip` package names are normalized, so `float16_sfogliatella` also resolves to the same distribution name.

If PyPI has not been published yet for the current version, install from source:

```bash
pip install "git+https://github.com/sigure-0424/Sfogliatella.git"
```

JAX GPU runtime is not pinned as a hard package dependency here. Install the matching JAX extra for your platform separately, for example:

```bash
pip install "jax[cuda12]"
```

## PyPI release flow

1. Bump `version` in `pyproject.toml`.
2. Create and publish a GitHub Release.
3. `.github/workflows/publish-pypi.yml` builds and publishes to PyPI via Trusted Publishing.

Before first publish, configure PyPI Trusted Publisher for this repo/workflow on PyPI.

## Quickstart

Python API:

```python
from sfogliatella import train, predict
```

CLI:

```bash
sfog-train --help
python -m train --help
```
