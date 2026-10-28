"""Run one method on one dataset and record its results.

Protocol (approval):
- hyperparameters tuned on the validation set, with comparable budgets for all methods
- results reported as mean +/- s.d. over 5 random seeds (training seeds 0-4, docs/open-questions.md #6)

Only the MLP receives the validation set (for early stopping); baselines are fit on the training
split alone (pass ``use_validation=True`` for the MLP). ``tune`` is the tuning step of
docs/open-questions.md #9: every configuration is fit with seed 0 and scored on the validation set.
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

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


def _val_score(model, X, y, metric):
    if metric == "accuracy":
        return float(accuracy_score(y, model.predict(X)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
    # roc_auc: probabilities if the model has them (an SVC fit without probability=True does not)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return float(roc_auc_score(y, scores))


def tune(make_model, configs, data, metric, use_validation=False):
    """Fit ``make_model(config, seed=0)`` for every config and score it on the validation set.

    ``metric`` is "accuracy" or "roc_auc" (higher is better), or "rmse" (lower is better); the
    test set is never touched. Returns ``(best_config, scores)`` with one validation score per config.
    """
    X_train, y_train, X_val, y_val = data[:4]
    scores = []
    for config in configs:
        model = make_model(config, 0)
        if use_validation:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        scores.append(_val_score(model, X_val, y_val, metric))
    pick = np.argmin if metric == "rmse" else np.argmax
    return configs[int(pick(scores))], scores


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
