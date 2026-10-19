"""Store and summarise experiment results (mean +/- s.d. over seeds) for tables in the report.

Format (docs/open-questions.md #12): one JSON per dataset x method in results/, e.g.
results/adult_rf.json, with keys ``dataset``, ``method``, ``hyperparameters``, ``seeds``,
``per_seed`` (one dict of metrics per seed) and ``mean_sd`` ({metric: [mean, sd]}).
The s.d. is the sample s.d. (ddof = 1) over the seeds. A metric that is None for any seed
(e.g. the HGB parameter count) is summarised as [None, None].
"""

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def summarise(runs):
    """Turn a list of per-seed result dicts into ``{name: [mean, sd]}`` (scalar entries only)."""
    out = {}
    for k, v in runs[0].items():
        if isinstance(v, (list, dict)):  # e.g. the confusion matrix
            continue
        values = [r[k] for r in runs]
        if any(x is None for x in values):
            out[k] = [None, None]
        else:
            out[k] = [float(np.mean(values)), float(np.std(values, ddof=1))]
    return out


def save(dataset, method, result):
    """Write ``result`` (from ``bench.runner.run``) to results/<dataset>_<method>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{dataset}_{method}.json"
    path.write_text(json.dumps({"dataset": dataset, "method": method, **result}, indent=2) + "\n")
    return path


def load_all():
    """All results/*.json, sorted by dataset then method."""
    results = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    return sorted(results, key=lambda r: (r["dataset"], r["method"]))
