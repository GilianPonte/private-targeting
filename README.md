![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Release](https://img.shields.io/github/v/release/GilianPonte/private-targeting)
![License](https://img.shields.io/github/license/GilianPonte/private-targeting)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GilianPonte/private-targeting/blob/main/examples/private_targeting_colab_demo.ipynb)

# Package for running the DP-CATE and DP-policy strategy,

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

There are two recommended ways to use the package:

1. **Google Colab**, for a quick browser-based tutorial.
2. **Local installation**, for running scripts on your own machine.

## Option 1: Install on Google Colab

Use this option if you want to try the package in a notebook without setting up a local Python environment.

Open a new Google Colab notebook and first check the Python version:

```python
import sys
print(sys.version)
```

The package is intended for Python **3.10 or 3.11**.

Install the released version from GitHub:

```python
%pip install --upgrade pip setuptools wheel
%pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"
```

After installation, restart the Colab runtime:

```text
Runtime -> Restart session
```

Then test the import in a new cell:

```python
from private_targeting import CTENN, DP_CATE, DP_policy

print("Import works")
```

A minimal Colab smoke test is:

```python
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
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

plt.figure(figsize=(6, 4))
plt.scatter(true_cate, cate_ctenn, alpha=0.6)
plt.axline((0, 0), slope=1, linestyle="--")
plt.xlabel("True CATE")
plt.ylabel("Estimated CATE from CTENN")
plt.title("CTENN CATE estimates")
plt.tight_layout()
plt.show()

summary = (
    policy_results
    .groupby(["percent", "epsilon"], as_index=False)
    .agg(mean_profit=("difference_from_random", "mean"))
)

plt.figure(figsize=(7, 4))

for eps, group in summary.groupby("epsilon"):
    group = group.sort_values("percent")
    plt.plot(group["percent"], group["mean_profit"], marker="o", label=str(eps))

plt.axhline(0, linestyle="--")
plt.xlabel("Targeted fraction")
plt.ylabel("Profit above random policy")
plt.title("DP-policy profit")
plt.legend(title="Policy")
plt.tight_layout()
plt.show()
```

## Option 2: Local installation

Use this option if you want to run the package on your own machine.

### Windows Command Prompt

```bat
py -3.11 -m venv private-targeting-env

private-targeting-env\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"
```

Check that the package imports correctly:

```bat
python -c "from private_targeting import CTENN, DP_CATE, DP_policy; print('Import works')"
```

### macOS or Linux

```bash
python3.11 -m venv private-targeting-env

source private-targeting-env/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"
```

Check that the package imports correctly:

```bash
python -c "from private_targeting import CTENN, DP_CATE, DP_policy; print('Import works')"
```

## Installation summary

| Use case                               | Recommended command                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Try in Google Colab                    | `%pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"` |
| Install locally on Windows/macOS/Linux | `pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"`  |
| Install latest GitHub version          | `pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git"`         |

## What the example does

The quick-start example demonstrates the three main functions in the package.

### `CTENN`

`CTENN` estimates conditional average treatment effects without privacy protection.

In the example, it uses:

```python
ate_ctenn, cate_ctenn, model_ctenn = CTENN(...)
```

The returned values are:

* `ate_ctenn`: the estimated average treatment effect.
* `cate_ctenn`: individual-level CATE estimates.
* `model_ctenn`: the fitted neural-network model used for the final CATE predictions.

This function is the non-private benchmark. It estimates treatment heterogeneity directly from the synthetic outcome, treatment, and feature data.

### `DP_CATE`

`DP_CATE` estimates conditional average treatment effects with differential privacy during model training.

In the example, it uses:

```python
ate_dp, cate_dp, model_dp, n_dp, epsilon, noise, epsilon_conservative = DP_CATE(...)
```

The returned values are:

* `ate_dp`: the differentially private estimated average treatment effect.
* `cate_dp`: differentially private individual-level CATE estimates.
* `model_dp`: the fitted private neural-network model.
* `n_dp`: the number of observations used for privacy accounting.
* `epsilon`: the estimated privacy-loss parameter.
* `noise`: the noise multiplier used during private optimization.
* `epsilon_conservative`: a conservative privacy-loss estimate.

The argument `noise_multiplier=1.0` controls how much noise is added during private training. Larger values usually imply stronger privacy (or lower epsilon from differential privacy) but noisier estimates.

The argument `fixed_model=True` is used in the tutorial to satisfy differential privacy.

### `DP_policy`

`DP_policy` evaluates targeting decisions under randomized-response privacy protection.

In the example, it uses:

```python
policy_results = DP_policy(...)
```

The function compares several targeting policies:

* `real`: oracle targeting using the true CATEs.
* `CTENN`: targeting using non-private CTENN estimates.
* one row for each privacy level in `epsilons`, such as `0.5`, `1`, and `3`;
* `random`: random targeting, used as the baseline.

The returned object is a pandas DataFrame with:

* `percent`: the targeted fraction of customers;
* `epsilon`: the policy or privacy level;
* `difference_from_random`: profit relative to random targeting;
* `iteration`: the simulation repetition.

In the quick-start example, `DP_policy` evaluates targeting the top 10% and 20% of customers over two repeated randomized runs.

## Requirements for the example

The example requires Python **3.10 or 3.11**.

The base package requires:

* `numpy`
* `pandas`
* `scikit-learn`

The full example also requires the machine-learning and plotting dependencies installed with:

```bash
pip install "private-targeting[full] @ git+https://github.com/GilianPonte/private-targeting.git@0.1.0"
```

The full installation includes TensorFlow, Keras Tuner, TensorFlow Privacy, TensorBoard, TensorFlow Probability, and Matplotlib.

The example uses a small synthetic dataset so that it runs quickly. It is intended as a smoke test to check that the package works, not as a full empirical replication of the paper.

## Fast tutorial with plots

After installing the package, the same example can be run as a script from the repository:

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

In Google Colab, TensorFlow may print dependency or runtime messages during installation. If imports fail immediately after installing, restart the runtime and rerun the import cell.

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
