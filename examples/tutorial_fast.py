"""Fast smoke-test tutorial for private-targeting.

Run from the repository root after installing with:

    pip install -e ".[dev,full]"
    python examples/tutorial_fast.py
"""

import os

# Keep TensorFlow logs quieter in small tutorial runs.
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

print("CTENN ATE:", ate_ctenn)

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

print("DP_CATE ATE:", ate_dp)
print("epsilon:", epsilon)
print("epsilon conservative:", epsilon_conservative)

policy_results = DP_policy(
    iterations=2,
    percentage=[0.10, 0.20],
    CATE=true_cate,
    CATE_estimates=cate_ctenn,
    epsilons=[0.5, 1, 3],
    seed_offset=1,
    verbose=False,
)

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
plt.title("DP policy profit")
plt.legend(title="Policy")
plt.tight_layout()
plt.show()
