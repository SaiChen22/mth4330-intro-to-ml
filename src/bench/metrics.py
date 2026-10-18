"""Evaluation metrics (approval).

- Fashion-MNIST: accuracy, macro-F1, confusion matrix                          (not done yet)
- Adult: ROC-AUC, F1 (positive class), balanced accuracy, log-loss              (not done yet)
- [x] California Housing: RMSE, MAE, R^2
- All datasets: training time, parameter count (measured by bench/runner.py)
"""

import numpy as np


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R^2 (same units as the target; $100k for California Housing)."""
    err = y_pred - y_true
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot),
    }
