# private-targeting

Python package for the training-stage private targeting estimator from the accompanying
academic paper.

## Current scope

This package turns the estimator code you shared into an installable Python package.
It currently includes the **CTENN / DP-CATE** estimator family:

- `cnn`: original non-private estimator name from your code
- `pcnn`: original private estimator name from your code
- `ctenn`: paper-aligned alias for `cnn`
- `dp_cate`: paper-aligned alias for `pcnn`

The **DP-policy** strategy from the paper is **not implemented in this package yet**,
because its code was not included in the snippet you provided.

## Installation

Base install:

```bash
pip install .
```

With machine-learning dependencies:

```bash
pip install .[full]
```

Editable install for development:

```bash
pip install -e .[dev,full]
```

## Quick start

```python
import numpy as np
from private_targeting import ctenn, dp_cate

n = 200
p = 10
rng = np.random.default_rng(42)
X = rng.normal(size=(n, p))
T = rng.binomial(1, 0.5, size=n)
Y = 0.5 * T + X[:, 0] + rng.normal(size=n)

ate, cate, model = ctenn(
    X,
    Y,
    T,
    epochs=5,
    max_epochs=2,
    folds=2,
    seed=42,
)

# Private version
# ate, cate, model, n, epsilon, noise_multiplier, epsilon_conservative = dp_cate(
#     X,
#     Y,
#     T,
#     epochs=5,
#     max_epochs=1,
#     batch_size=50,
#     noise_multiplier=1.0,
#     seed=42,
# )
```

## Project layout

```text
private_targeting_pkg/
├── pyproject.toml
├── README.md
├── CITATION.cff
├── src/
│   └── private_targeting/
│       ├── __init__.py
│       └── dp_cate.py
└── tests/
    └── test_api.py
```

## Notes on the packaged code

The package preserves the structure of your original implementation, while fixing a few
issues that would otherwise break packaging or runtime behavior:

- makes the project installable with a `src/` layout
- fixes the undefined `noise_multiplier` filename reference in `cnn`
- validates that `T` is binary
- adds a clearer error for unstable pseudo-outcomes when residualized treatment is near zero
- replaces the invalid Keras activation string `"leaky_relu"` with an explicit layer
- keeps TensorFlow, Keras Tuner, and TensorFlow Privacy imports lazy inside the estimators

## Before publishing

You should still update these fields before making the repository public:

- author names in `pyproject.toml`
- GitHub URLs in `pyproject.toml`
- license choice
- version number

## Running tests

```bash
pytest
```
