"""Run one method on one dataset and record its results.

Protocol (approval):
- hyperparameters tuned on the validation set, with comparable budgets for all methods
- results reported as mean +/- s.d. over 5 random seeds (training seeds 0-4, docs/open-questions.md #6)

Only the MLP receives the validation set (for early stopping); baselines are fit on the training
split alone (pass ``use_validation=True`` for the MLP). The tuning loop of docs/open-questions.md #9
is not implemented yet: the caller passes the chosen model and records its hyperparameters.
"""

import time

import numpy as np

from .metrics import confusion_matrix
from .results import summarise


def n_parameters(model):
    """Parameter count by the rule of docs/open-questions.md #12 (None means n/a).

    Linear models and our MLP: weights and biases. Random forest: total tree nodes.
    SVM: support vectors. Anything else (HGB): None. For trees and kernel methods this is only a
    proxy for model size and is not comparable with a neural network.
    """
    if hasattr(model, "n_parameters"):  # our MLP
        return int(model.n_parameters())
    if hasattr(model, "coef_"):  # scikit-learn linear models
        return int(np.size(model.coef_) + np.size(model.intercept_))
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "tree_"):  # random forest
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    if hasattr(model, "support_"):  # SVC / SVR
        return int(model.support_.size)
    return None


def run(make_model, data, metrics_fn, seeds=range(5), use_validation=False, use_proba=False,
        hyperparameters=None, n_classes=None):
    """Fit ``make_model(seed)`` for every seed; return a result dict for ``results.save``.

    ``data`` is what a loader returns: ``X_train, y_train, X_val, y_val, X_test, y_test``.
    ``metrics_fn(y_true, y_pred)`` is called on the test set, or ``metrics_fn(y_true, y_pred, proba)``
    with ``use_proba=True`` (Adult). Per seed we also record ``train_time`` (seconds),
    ``n_parameters`` and, if ``n_classes`` is given, the test ``confusion_matrix`` (nested lists).
    ``hyperparameters`` is a JSON-serialisable dict stored as is.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = data
    per_seed = []
    for seed in seeds:
        model = make_model(seed)
        start = time.perf_counter()
        if use_validation:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        y_pred = model.predict(X_test)
        extra = (model.predict_proba(X_test),) if use_proba else ()
        record = {**metrics_fn(y_test, y_pred, *extra), "train_time": elapsed, "n_parameters": n_parameters(model)}
        if n_classes is not None:
            record["confusion_matrix"] = confusion_matrix(y_test, y_pred, n_classes).tolist()
        per_seed.append(record)
    return {
        "hyperparameters": hyperparameters or {},
        "seeds": list(seeds),
        "per_seed": per_seed,
        "mean_sd": summarise(per_seed),
    }
