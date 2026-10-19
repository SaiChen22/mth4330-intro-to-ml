"""Print a table of every results/*.json (mean ± s.d. over seeds), grouped by dataset.

Run: python scripts/summarize_results.py
"""

from bench.results import load_all


def fmt(mean_sd):
    mean, sd = mean_sd
    return "n/a" if mean is None else f"{mean:.4g}±{sd:.2g}"


results = load_all()
for dataset in sorted({r["dataset"] for r in results}):
    print(f"\n== {dataset}")
    for r in (r for r in results if r["dataset"] == dataset):
        print(f"{r['method']:12s}" + "  ".join(f"{k} {fmt(v)}" for k, v in r["mean_sd"].items()))
print("\nn_parameters: weights+biases (linear, MLP), tree nodes (random forest), support vectors (SVM), "
      "n/a (HGB). For trees and kernel methods it only proxies model size and is not comparable "
      "with a neural network.")
