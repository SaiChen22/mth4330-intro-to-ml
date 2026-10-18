"""Store and summarise experiment results (mean +/- s.d. over seeds) for tables in the report.

Format of stored results: TO DECIDE (docs/open-questions.md #12). For now only the in-memory
summary exists; the s.d. is the sample s.d. (ddof = 1) over the seeds.
"""

import numpy as np


def summarise(runs):
    """Turn a list of per-seed result dicts into ``{name: (mean, sd)}``."""
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1))) for k in runs[0]}
