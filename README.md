# float16-sfogliatella

Sfogliatella is a general-purpose time-series machine learning library with Python API and CLI workflows for training, prediction, live inference, HPO, stacking, and model packaging.

## Install

CPU/default install:

```bash
pip install float16-sfogliatella
```

JAX GPU runtime is not pinned as a hard package dependency here. Install the matching JAX extra for your platform separately, for example:

```bash
pip install "jax[cuda12]"
```

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
