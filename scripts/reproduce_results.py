"""Reproduce every result reported in the report.

Guideline: the repository must contain commands that reproduce the reported results.
Run: python scripts/reproduce_results.py   (after scripts/download_data.py)

So far: the preliminary-report pilot on California Housing, ridge baseline vs. our MLP, seeds 0-4,
split seed 4330. Hyperparameters are fixed and untuned (the search space is
docs/open-questions.md #9). Test metrics are in units of $100k.
Results go to results/<dataset>_<method>.json; print the tables with scripts/summarize_results.py.
"""

from baselines.linear import ridge
from bench.data.housing import load_housing
from bench.metrics import regression_metrics
from bench.results import save
from bench.runner import run
from mlp.model import MLP

data = load_housing()

MLP_PARAMS = dict(
    hidden_sizes=[64, 64], activation="relu", task="regression", init="he",
    weight_decay=0.0, dropout=0.0, optimizer="adam", lr=1e-3,
    batch_size=128, epochs=50, patience=5,
)

for method, make, hyperparameters, use_val in [
    ("ridge", lambda seed: ridge(1.0), {"alpha": 1.0}, False),
    ("mlp", lambda seed: MLP(**MLP_PARAMS, seed=seed), MLP_PARAMS, True),
]:
    result = run(make, data, regression_metrics, use_validation=use_val, hyperparameters=hyperparameters)
    print("saved", save("housing", method, result))
