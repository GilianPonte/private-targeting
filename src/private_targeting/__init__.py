"""Private targeting estimators for the accompanying academic paper.

This package currently exposes the training-stage estimator from the paper via both the
original function names (``cnn``, ``pcnn``) and paper-aligned aliases (``ctenn``,
``dp_cate``).
"""

from .dp_cate import cnn, ctenn, dp_cate, pcnn

__version__ = "0.1.0"

__all__ = ["cnn", "pcnn", "ctenn", "dp_cate", "__version__"]
