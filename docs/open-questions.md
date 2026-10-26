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
| 8 | RBF-SVM: size of the training subsample on Fashion-MNIST (and Adult?) | Fashion-MNIST: stratified 10,000 of the 54k training set (sampling seed 4330). Adult, Housing: full training set first; see [SVM subsample](#8-svm-subsample) below. | 2026-10-18 |
| 9 | Hyperparameter search space and budget for each method; validation metric used for selection | Selection metric on validation: accuracy (F-MNIST), ROC-AUC (Adult), RMSE (Housing). 3–6 configs per method and dataset, tuned with seed 0, then the best config rerun with seeds 0–4. Grid values: pending (written by the baselines owner, reviewed by Sai). See [Tuning](#9-tuning) below. | 2026-10-18 |
| **Code** | | | |
| 10 | Public interface of the MLP (class/function names, arguments, return values) | scikit-learn style `MLP` class; full specification in [MLP interface](#10-mlp-interface) below. | 2026-10-18 |
| 11 | Interface of the dataset loaders (what they return) | `load_xxx(split_seed=4330)` returns six NumPy arrays; see [Dataset loaders](#11-dataset-loaders) below. | 2026-10-18 |
| 12 | Result file format and where results are stored | One JSON per dataset × method in `results/` (e.g. `results/adult_rf.json`), plus a small summary script that reads them all; see [Results files](#12-results-files-and-parameter-counts) below. | 2026-10-18 |
| 13 | Python version; pinned dependency versions? | Python >= 3.11 (tested on 3.11 and on Colab's current Python). Versions pinned in `requirements.txt`. Checked 2026-10-18: the pins install and all tests pass on Python 3.11.16 (numpy is pinned to 2.4.6 because 2.5 needs Python >= 3.12); Colab not tested yet. | 2026-10-18 |
| 14 | Test framework (e.g. pytest)? Continuous integration on GitHub? | pytest. No CI for now. | 2026-10-18 |
| 15 | Packaging / import path for `src/` (e.g. `pip install -e .` vs. `PYTHONPATH`) | Add a `pyproject.toml`; `pip install -e .` makes `mlp`, `bench`, `baselines` and `reference` importable. | 2026-10-18 |
| **Repository and writing** | | | |
| 16 | Default branch name; branch / pull-request / review rules | Default branch stays `master`; one branch per person from the latest `master`; Sai merges, no PR review for now; no direct commits to `master`, no force push. See [Branches](#16-branches) below. | 2026-10-18 |
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

### 3 and 7. Fashion-MNIST (source and dtype, decided 2026-10-18)

- Source: the official IDX files from Zalando's repository (this fixes the official 60k / 10k split).
- Pixels are divided by 255 and stored as float32 (784 features per image). The MLP casts them to
  its own `dtype`.

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

### Layer design (decided 2026-10-18)

- `src/mlp/layers.py` holds two classes, `Dense` and `Dropout`, each with `forward` and `backward`.
- `MLP` in `src/mlp/model.py` chains them.

### 8. SVM subsample

- Fashion-MNIST: a stratified 10,000 of the 54k training images, sampling seed 4330, the same subset
  for every training seed.
- Adult and Housing: the full training set first. If one Adult fit takes more than about 5 minutes,
  use the same kind of stratified 10k subset and state in the report that this goes beyond the
  approval.
- The report must say that the SVM used a subsample.

### 9. Tuning

- Selection metric, on the validation set only: accuracy (Fashion-MNIST), ROC-AUC (Adult),
  RMSE (Housing).
- Budget: 3–6 configurations per method and dataset, roughly the same number for every method,
  tuning 1–2 main hyperparameters per method.
- Procedure: (1) train every configuration with seed 0 and score it on the validation set;
  (2) pick the best one; (3) rerun that configuration with seeds 0–4 and report the test-set
  mean ± s.d.
- Grid shapes: ridge `alpha` and logistic `C`, 3 values each; random forest `max_depth`, 3 values;
  HGB `learning_rate`, 3 values; SVM `C`, 2 values; MLP width and learning rate, 2 values each
  (4 configurations).
- The baselines owner lists the concrete values; Sai reviews them before they are written into this
  file. Pending.
- SVM probabilities: not enabled during tuning; the final fit uses `probability=True` and
  `random_state=seed`. The report must say that for a probability-enabled SVC the s.d. is not
  strictly 0 and that `predict` and the argmax of the probabilities occasionally disagree.

### 12. Results files and parameter counts

- One JSON per dataset × method in `results/`, e.g. `results/adult_rf.json`. Content: the selected
  hyperparameters, per-seed metrics, and mean ± s.d.; for Fashion-MNIST the confusion matrix as
  nested lists.
- A small summary script reads all the JSON files and prints the tables. Separate files mean parallel
  experiments never write to the same file.
- Parameter count: linear models and the MLP, the actual number of weights and biases; random
  forest, the total number of tree nodes; SVM, the number of support vectors; HGB, `n/a`. A table
  footnote must say that for trees and kernel methods the count only proxies model size and cannot
  be compared directly with neural networks.

### 16. Branches

- Default branch stays `master`.
- Everyone works on their own branch from the latest `master`, e.g. `feat/mlp-core`,
  `feat/baselines`.
- Sai merges; no PR review for now.
- No direct commits to `master`, no force push.

### Implementation choices (not fixed by the approval, recorded 2026-10-18)

- **Initialisation:** Xavier is the uniform version, W_ij ~ U(-a, a) with a = sqrt(6 / (fan_in + fan_out));
  He is the normal version, W_ij ~ N(0, 2 / fan_in). Biases start at zero.
- **Weight layout:** W has shape (fan_out, fan_in), as in the report's z = W h + b. Data is stored one
  example per row, so a layer computes Z = H W^T + b.
- **Optimisers:** momentum uses v <- mu v + g, p <- p - lr v. Adam uses the standard bias-corrected
  update with beta = (0.9, 0.999) and eps = 1e-8.
- **MLP constructor:** all arguments are keyword-only. `weight_decay`, `dropout` and `momentum`
  default to 0 and `dtype` to float64; every other argument is required.
- **`history_`:** a dict `{"train": [...], "val": [...]}` with one entry per epoch. The training loss
  is computed at the end of each epoch on the full training set in prediction mode (no dropout);
  `"val"` stays empty when no validation data is passed.
- **Predictions:** for classification `predict` returns the argmax of the output; for Adult
  (2-output softmax) this is the 0.5 threshold on the positive-class probability.
- **Numerics:** softmax, cross-entropy and sigmoid are computed in an overflow-safe way (shift by the
  row maximum; exponent of a non-positive number only).
- **Seeds summary:** the s.d. over the 5 seeds is the sample s.d. (ddof = 1).
- **Data download:** `scripts/download_data.py` downloads the raw files into `data/`; the loaders
  only read them and raise an error that points to the script if a file is missing.
