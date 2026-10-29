"""Baseline methods (scikit-learn), kept separate from our own code in src/mlp/.

Planned baselines (approval):
- [x] softmax / logistic regression (classification) and ridge regression (regression)
      = the no-hidden-layer MLP
- [x] random forest
- [x] histogram gradient boosting
- [-] RBF-kernel SVM / SVR: dropped on 2026-10-28 (docs/open-questions.md #20)

Guideline: same splits, preprocessing and metrics as our MLP; comparable tuning effort.
"""
