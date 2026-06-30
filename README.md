# DP-CATE and DP-policy

Python package for private targeting with `CTENN`, `DP_CATE`, and `DP_policy` from the paper:

> Ponte, Gilian R., Tom Boot, Thomas Reutterer, and Jaap E. Wieringa. “EXPRESS: Where Should Firms Implement Differential Privacy in Targeting? Implications for Profitability.” *Journal of Marketing Research*, 2026. DOI: 10.1177/00222437261455302.

## Current scope

This package exposes three public functions:

* `CTENN`: non-private CATE estimator.
* `DP_CATE`: differentially private CATE estimator.
* `DP_policy`: randomized-response targeting policy evaluation.

## Python version

Use Python **3.10 or 3.11**.

The full machine-learning stack depends on `tensorflow-privacy==0.9.0`, which requires Python `<3.12`. Python 3.11 is recommended.

## Installation

### Install from GitHub release

To install the released version directly from GitHub:

```bash
pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@v0.1.0"
```

### Install from the latest GitHub main branch

```bash
pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git"
```

### Local development install

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/GilianPonte/private-targeting.git
cd private-targeting
pip install -e ".[dev,full]"
```

## Clean local setup on Windows Command Prompt

```bat
git clone https://github.com/GilianPonte/private-targeting.git
cd private-targeting

py -3.11 -m venv .venv

.venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

pip install -e ".[dev,full]"
```

Check that the package imports correctly:

```bat
python -c "from private_targeting import CTENN, DP_CATE, DP_policy; print('Import works')"
```

## Clean local setup on macOS or Linux

```bash
git clone https://github.com/GilianPonte/private-targeting.git
cd private-targeting

python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install -e ".[dev,full]"
```

Check that the package imports correctly:

```bash
python -c "from private_targeting import CTENN, DP_CATE, DP_policy; print('Import works')"
```

## Quick start

```python
import numpy as np

from private_targeting import CTENN, DP_CATE, DP_policy

rng = np.random.default_rng(1)

n = 100
p = 5

X = rng.normal(size=(n, p))
T = rng.binomial(1, 0.5, size=n)

true_cate = 1 + 0.5 * X[:, 0] - 0.25 * X[:, 1]
Y = 2 + X[:, 0] + X[:, 1] + T * true_cate + rng.normal(scale=1, size=n)

ate_ctenn, cate_ctenn, model_ctenn = CTENN(
    X=X,
    Y=Y,
    T=T,
    folds=2,
    epochs=1,
    max_epochs=1,
    batch_size=10,
    seed=1,
)

ate_dp, cate_dp, model_dp, n_dp, epsilon, noise, epsilon_conservative = DP_CATE(
    X=X,
    Y=Y,
    T=T,
    epochs=1,
    max_epochs=1,
    batch_size=10,
    noise_multiplier=1.0,
    fixed_model=True,
    seed=1,
)

policy_results = DP_policy(
    iterations=2,
    percentage=[0.10, 0.20],
    CATE=true_cate,
    CATE_estimates=cate_ctenn,
    epsilons=[0.5, 1, 3],
    seed_offset=1,
    verbose=False,
)

print("CTENN ATE:", ate_ctenn)
print("DP_CATE ATE:", ate_dp)
print(policy_results)
```

## Fast tutorial with plots

After installing with `.[dev,full]`, run:

```bash
python examples/tutorial_fast.py
```

On Windows Command Prompt:

```bat
python examples\tutorial_fast.py
```

The tutorial creates a toy dataset, estimates CATEs with `CTENN` and `DP_CATE`, evaluates protected targeting with `DP_policy`, and opens two plots:

1. estimated versus true CATEs;
2. DP-policy profit above random targeting.

The tutorial is designed as a fast smoke test. It uses very small training settings, so it is not intended to reproduce the empirical results from the paper.

## Project layout

```text
private-targeting/
├── pyproject.toml
├── README.md
├── examples/
│   └── tutorial_fast.py
├── src/
│   └── private_targeting/
│       ├── __init__.py
│       └── dp_cate.py
└── tests/
    └── test_api.py
```

## Notes

TensorFlow on native Windows prints a warning that GPU support is not available for TensorFlow >= 2.11. That warning is expected; the fast tutorial runs on CPU.

TensorFlow may also print oneDNN or CPU optimization messages. These are informational messages, not errors.

To reduce TensorFlow log output on Windows Command Prompt, run:

```bat
set TF_CPP_MIN_LOG_LEVEL=2
```

Then rerun the tutorial:

```bat
python examples\tutorial_fast.py
```

## Citation

If you use this package, please cite:

```bibtex
@article{ponte2026differentialprivacytargeting,
  title   = {EXPRESS: Where Should Firms Implement Differential Privacy in Targeting? Implications for Profitability},
  author  = {Ponte, Gilian R. and Boot, Tom and Reutterer, Thomas and Wieringa, Jaap E.},
  journal = {Journal of Marketing Research},
  year    = {2026},
  doi     = {10.1177/00222437261455302}
}
```
