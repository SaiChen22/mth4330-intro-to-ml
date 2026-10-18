"""Pilot on California Housing: ridge baseline vs. our MLP, seeds 0-4, split seed 4330.

Hyperparameters are fixed and untuned (pilot only; the search space is docs/open-questions.md #9).
Test metrics are in units of $100k. Run: python scripts/pilot_housing.py
"""

from baselines.linear import ridge
from bench.data.housing import load_housing
from bench.metrics import regression_metrics
from bench.runner import run
from mlp.model import MLP

data = load_housing()


def make_mlp(seed):
    return MLP(
        hidden_sizes=(64, 64), activation="relu", task="regression", init="he",
        weight_decay=0.0, dropout=0.0, optimizer="adam", lr=1e-3,
        batch_size=128, epochs=50, patience=5, seed=seed,
    )


for name, make, use_val in [
    ("ridge (alpha=1)", lambda seed: ridge(1.0), False),
    ("MLP (64,64) relu adam", make_mlp, True),
]:
    summary = run(make, data, regression_metrics, use_validation=use_val)
    print(f"{name:24s}" + "  ".join(f"{k} {m:.4g}±{s:.2g}" for k, (m, s) in summary.items()))
