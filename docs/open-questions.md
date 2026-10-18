# Open questions

Decisions the approved proposal does not settle. Decide them as a team, record the answer here
(with the date), and only then write them into code or docs. Changes to the approved plan must also
be reported in the Introduction of the preliminary report.

| # | Question | Decision | Date |
|---|---|---|---|
| **Team** | | | |
| 1 | Who owns which part (MLP code, data/baselines, math/writing, experiments)? | | |
| 2 | Internal schedule between now and the final report | | |
| **Data and protocol** | | | |
| 3 | Fashion-MNIST: how is the validation set taken from the official training set (size, stratified or not)? | Stratified 10% of the official 60k training set is validation (54k train / 6k validation). The official 10k set is the test set. | 2026-10-18 |
| 4 | Adult: use UCI's two files (`adult.data`, `adult.test`) pooled, or keep their split? How are missing values (`?`) handled? Keep all 14 features? | Pool both files (48,842 records); see [Adult](#4-adult) below. `?` becomes its own `Unknown` category. Keep all 14 features. | 2026-10-18 |
| 5 | California Housing: how to stratify a continuous target for the 70/15/15 split? Scale the target? | Stratify on 10 quantile bins of the target. Target is not scaled for any method (units of $100k). See [California Housing](#5-california-housing) below. | 2026-10-18 |
| 6 | Random seed(s) that fix the splits; which 5 seeds are used for training randomness | Split seed 4330, shared by all methods. Training randomness uses seeds 0–4 for every method, including `random_state` of RF / HGB. | 2026-10-18 |
| 7 | Fashion-MNIST pixel preprocessing (e.g. divide by 255, standardise) | Divide by 255 and flatten to 784 features. No other preprocessing. | 2026-10-18 |
| 8 | RBF-SVM: size of the training subsample on Fashion-MNIST (and Adult?) | | |
| 9 | Hyperparameter search space and budget for each method; validation metric used for selection | | |
| **Code** | | | |
| 10 | Public interface of the MLP (class/function names, arguments, return values) | scikit-learn style `MLP` class; full specification in [MLP interface](#10-mlp-interface) below. | 2026-10-18 |
| 11 | Interface of the dataset loaders (what they return) | `load_xxx(split_seed=4330)` returns six NumPy arrays; see [Dataset loaders](#11-dataset-loaders) below. | 2026-10-18 |
| 12 | Result file format and where results are stored | | |
| 13 | Python version; pinned dependency versions? | Python >= 3.11 (tested on 3.11 and on Colab's current Python). Versions pinned in `requirements.txt`. | 2026-10-18 |
| 14 | Test framework (e.g. pytest)? Continuous integration on GitHub? | pytest. No CI for now. | 2026-10-18 |
| 15 | Packaging / import path for `src/` (e.g. `pip install -e .` vs. `PYTHONPATH`) | Add a `pyproject.toml`; `pip install -e .` makes `mlp`, `bench`, `baselines` and `reference` importable. | 2026-10-18 |
| **Repository and writing** | | | |
| 16 | Default branch name; branch / pull-request / review rules | | |
| 17 | License for the code | | |
| 18 | Report: one `main.tex` or one file per section? Overleaf or the repo as the source of truth? | | |
| 19 | Shared notation for the report (symbols, vector/matrix conventions) | | |

## Details of the decisions

### 4. Adult

- Pool `adult.data` and `adult.test`. For `adult.test`, skip the first line and strip the trailing `.`
  from the labels. Strip leading spaces from all values.
- Stratified 70/15/15 split on the income label.
- `?` occurs only in categorical columns, so it becomes its own `Unknown` category.
- Keep all 14 features.
- Standardise the numeric columns and one-hot encode the categorical ones (unseen categories are
  ignored). Both are fit on the training split only.

### 5. California Housing

- Bin the target into 10 quantile bins for a stratified 70/15/15 split.
- Standardise the 8 features (fit on the training split only).
- The target is left unscaled for every method. scikit-learn's version is already in units of $100k
  (mean about 2.07, s.d. about 1.15), so RMSE and MAE are reported in $100k units and nothing needs to
  be inverted.

### 10. MLP interface

Scikit-learn style, so the runner can treat the MLP and the baselines the same way. The one
difference: only the MLP receives the validation set (for early stopping). This must be stated in
the report.

```
MLP(hidden_sizes, activation, task, init, weight_decay, dropout, optimizer, lr, momentum,
    batch_size, epochs, patience, seed, dtype)
```

**Arguments**
- `hidden_sizes` excludes the input and output layers, e.g. `(128, 64)`. `()` means zero hidden
  layers (softmax / ridge regression). The input size and the number of classes are inferred in
  `fit` (`k = y_train.max() + 1`).
- `activation` in {`relu`, `tanh`, `sigmoid`}; `init` in {`xavier`, `he`};
  `optimizer` in {`sgd`, `momentum`, `adam`}.
- `momentum` is only used when `optimizer="momentum"`. Adam uses the default beta = (0.9, 0.999) and
  eps = 1e-8, which are not exposed as parameters.
- `seed` controls the initialisation, the shuffling and the dropout.
- `task` in {`"classification"`, `"regression"`}. Binary classification (Adult) uses a 2-output
  softmax, which stays within the approved scope.
- `dtype` defaults to float64 (gradient checks); float32 is allowed for large runs.
- `epochs` is the maximum number of epochs.

**Methods**
- `.fit(X, y, X_val=None, y_val=None) -> self`
- `.predict(X)`
- `.predict_proba(X)` (classification only)
- `.n_parameters()`

**Labels:** integers `0..k-1` for classification; a float array of shape `(n,)` for regression.

**Loss:** mean of 1/2 (y_hat - y)^2 (regression) or cross-entropy (classification), plus
(lambda/2) * sum of ||W||^2 over the weights. Biases are not penalised. The L2 term is not rescaled
by the batch size.

**Dropout:** `dropout` is the drop probability; inverted dropout, applied to hidden layers only.

**Early stopping:** monitors the validation data loss (without the L2 term) and restores the best
weights. Training stops after `patience` epochs without an improvement (no `min_delta`). Early
stopping is off when no validation data is passed.

**Diagnostics**
- Attribute `history_` records the train and validation data loss (without the L2 term) per epoch;
  for regression this is half the MSE.
- Option `record_norms=False`: when enabled, records per-layer activation and gradient norms (needed
  for the gradient-flow experiment; not needed for Oct 21).

**Zero-hidden-layer check against scikit-learn:** `Ridge(alpha = n * lambda)`,
`LogisticRegression(C = 1 / (n * lambda))`.

### 11. Dataset loaders

- `load_xxx(split_seed=4330)` returns `X_train, y_train, X_val, y_val, X_test, y_test` as NumPy
  arrays.
- All feature preprocessing, fit on the training split only, happens inside the loader, so every
  method sees identical data.
- Targets are returned as they are (class indices for classification, raw values for regression).
