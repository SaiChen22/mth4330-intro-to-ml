"""Evaluation metrics (approval).

- [x] Fashion-MNIST: accuracy, macro-F1, confusion matrix
- [x] Adult: ROC-AUC, F1 (positive class), balanced accuracy, log-loss
- [x] California Housing: RMSE, MAE, R^2
- All datasets: training time, parameter count (measured by bench/runner.py)

Scalar metrics are returned as dicts of floats (summarised over seeds by bench/results.py); the
confusion matrix is a separate function because it is not a scalar.
"""

import numpy as np
from sklearn import metrics as skm


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


def multiclass_metrics(y_true, y_pred):
    """Accuracy and macro-averaged F1 (Fashion-MNIST)."""
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "macro_f1": float(skm.f1_score(y_true, y_pred, average="macro")),
    }


def confusion_matrix(y_true, y_pred, n_classes=10):
    """Counts with rows = true class, columns = predicted class."""
    return skm.confusion_matrix(y_true, y_pred, labels=range(n_classes))


def binary_metrics(y_true, y_pred, proba):
    """ROC-AUC, F1 of the positive class (label 1), balanced accuracy and log-loss (Adult).

    ``proba`` has shape (n, 2); the predicted label is ``y_pred`` (threshold 0.5 / argmax).
    """
    return {
        "roc_auc": float(skm.roc_auc_score(y_true, proba[:, 1])),
        "f1": float(skm.f1_score(y_true, y_pred, pos_label=1)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(skm.log_loss(y_true, proba, labels=[0, 1])),
    }
