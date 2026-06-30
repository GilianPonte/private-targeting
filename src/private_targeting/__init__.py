"""Private targeting estimators for the accompanying academic paper.

This package currently exposes the training-stage estimator from the paper via both the
original function names (``cnn``, ``pcnn``) and paper-aligned aliases (``ctenn``,
``dp_cate``).
"""

from .estimators import CTENN, DP_CATE, DP_policy

__all__ = ["CTENN", "DP_CATE", "DP_policy"]
