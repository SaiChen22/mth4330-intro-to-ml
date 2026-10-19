"""Run one method on one dataset and record its results.

Protocol (approval):
- hyperparameters tuned on the validation set, with comparable budgets for all methods
- results reported as mean +/- s.d. over 5 random seeds (training seeds 0-4, docs/open-questions.md #6)

Only the MLP receives the validation set (for early stopping); baselines are fit on the training
split alone (pass ``use_validation=True`` for the MLP). Hyperparameter tuning is not implemented
yet (docs/open-questions.md #9): the caller passes a fixed model.
"""

import time

from .results import summarise


def n_parameters(model):
    if hasattr(model, "n_parameters"):  # our MLP
        return model.n_parameters()
    return model.coef_.size + model.intercept_.size  # scikit-learn linear models


def run(make_model, data, metrics_fn, seeds=range(5), use_validation=False, use_proba=False):
    """Fit ``make_model(seed)`` for every seed; return the summary ``{name: (mean, sd)}``.

    ``data`` is what a loader returns: ``X_train, y_train, X_val, y_val, X_test, y_test``.
    ``metrics_fn(y_true, y_pred)`` is called on the test set, or ``metrics_fn(y_true, y_pred, proba)``
    with ``use_proba=True`` (Adult). ``train_time`` (seconds) and ``n_parameters`` are recorded per seed.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = data
    runs = []
    for seed in seeds:
        model = make_model(seed)
        start = time.perf_counter()
        if use_validation:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        extra = (model.predict_proba(X_test),) if use_proba else ()
        scores = metrics_fn(y_test, model.predict(X_test), *extra)
        runs.append({**scores, "train_time": elapsed, "n_parameters": n_parameters(model)})
    return summarise(runs)
