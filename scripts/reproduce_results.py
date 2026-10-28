"""Reproduce every result reported in the report.

Guideline: the repository must contain commands that reproduce the reported results.
Run (after scripts/download_data.py):

    python scripts/reproduce_results.py                      # all datasets, all methods
    python scripts/reproduce_results.py adult --methods rf mlp

Protocol (docs/open-questions.md #6, #8, #9): for every dataset and method, each configuration of
the grid below is fit with seed 0 and scored on the validation set (accuracy for Fashion-MNIST,
ROC-AUC for Adult, RMSE for Housing); the best one is rerun with seeds 0-4 and scored on the
test set. Only the MLP gets the validation set during fitting (early stopping). The SVM uses a
stratified 10,000-image subsample of the Fashion-MNIST training set (sampling seed 4330).
Results go to results/<dataset>_<method>.json; print the tables with scripts/summarize_results.py.

The grid VALUES are proposed and still pending Sai's review (#9 lists only their shapes).
"""

import argparse
import time

from baselines.hist_gradient_boosting import hist_gradient_boosting
from baselines.linear import ridge
from baselines.logistic import logistic
from baselines.random_forest import random_forest
from baselines.SVM import svm
from bench.data.adult import load_adult
from bench.data.fmnist import load_fmnist
from bench.data.housing import load_housing
from bench.metrics import binary_metrics, multiclass_metrics, regression_metrics
from bench.results import save
from bench.runner import run, tune
from bench.splits import stratified_holdout
from mlp.model import MLP

SEEDS = range(5)
SVM_SUBSAMPLE = 10_000

# dataset: loader, task, validation metric for selection, test metrics, probabilities needed, classes
DATASETS = {
    "housing": (load_housing, "regression", "rmse", regression_metrics, False, None),
    "adult": (load_adult, "classification", "roc_auc", binary_metrics, True, None),
    "fmnist": (load_fmnist, "classification", "accuracy", multiclass_metrics, False, 10),
}

# One grid per method: a list of configurations (3 values, SVM 2, MLP 2 x 2).
LINEAR_GRID = {"regression": [{"alpha": a} for a in (0.01, 1, 100)],
               "classification": [{"C": c} for c in (0.01, 0.1, 1)]}
RF_GRID = [{"max_depth": d} for d in (5, 10, 20)]
HGB_GRID = [{"learning_rate": lr} for lr in (0.03, 0.1, 0.3)]
SVM_GRID = [{"C": c} for c in (1, 10)]
MLP_GRID = [{"width": w, "lr": lr} for w in (64, 128) for lr in (1e-3, 1e-2)]
MLP_FIXED = dict(activation="relu", init="he", weight_decay=0.0, dropout=0.0, optimizer="adam",
                 batch_size=128, epochs=50, patience=5)


def methods(dataset, task, needs_proba):
    """{name: (make(config, seed, final), grid, use_validation)} for one dataset."""
    linear = (lambda c, s, final: ridge(c["alpha"])) if task == "regression" else (lambda c, s, final: logistic(c["C"], s))
    float_type = "float32" if dataset == "fmnist" else "float64"  # float32 allowed for large runs (#10)
    return {
        "ridge" if task == "regression" else "logistic": (linear, LINEAR_GRID[task], False),
        "rf": (lambda c, s, final: random_forest(task, c["max_depth"], s), RF_GRID, False),
        "hgb": (lambda c, s, final: hist_gradient_boosting(task, c["learning_rate"], s), HGB_GRID, False),
        # SVM probabilities only for the final fit, and only where the metrics need them (Adult)
        "svm": (lambda c, s, final: svm(task, c["C"], s, probability=final and needs_proba), SVM_GRID, False),
        "mlp": (lambda c, s, final: MLP(hidden_sizes=(c["width"],) * 2, task=task, lr=c["lr"], seed=s,
                                        dtype=float_type, **MLP_FIXED), MLP_GRID, True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--methods", nargs="*", help="subset of: ridge logistic rf hgb svm mlp")
    args = parser.parse_args()

    for dataset in args.datasets:
        load, task, select, metrics_fn, needs_proba, n_classes = DATASETS[dataset]
        data = load()
        for name, (make, grid, use_val) in methods(dataset, task, needs_proba).items():
            if args.methods and name not in args.methods:
                continue
            start = time.perf_counter()
            fit_data, extra = data, {}
            if name == "svm" and dataset == "fmnist":
                keep = stratified_holdout(data[1], SVM_SUBSAMPLE, 4330)[1]
                fit_data = (data[0][keep], data[1][keep], *data[2:])
                extra = {"train_subsample": SVM_SUBSAMPLE}
            best, scores = tune(lambda c, s: make(c, s, False), grid, fit_data, select, use_val)
            result = run(lambda s: make(best, s, True), fit_data, metrics_fn, SEEDS, use_val,
                         use_proba=needs_proba, n_classes=n_classes,
                         hyperparameters={"selected": best, **(MLP_FIXED if name == "mlp" else {}), **extra,
                                          "tuning": {"metric": select, "configs": grid, "val_scores": scores}})
            print(f"{dataset:8s}{name:10s}{best}  {time.perf_counter() - start:6.0f}s  -> {save(dataset, name, result)}", flush=True)


if __name__ == "__main__":
    main()
