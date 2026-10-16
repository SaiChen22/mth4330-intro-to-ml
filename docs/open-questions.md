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
| 3 | Fashion-MNIST: how is the validation set taken from the official training set (size, stratified or not)? | | |
| 4 | Adult: use UCI's two files (`adult.data`, `adult.test`) pooled, or keep their split? How are missing values (`?`) handled? Keep all 14 features? | | |
| 5 | California Housing: how to stratify a continuous target for the 70/15/15 split? Scale the target? | | |
| 6 | Random seed(s) that fix the splits; which 5 seeds are used for training randomness | | |
| 7 | Fashion-MNIST pixel preprocessing (e.g. divide by 255, standardise) | | |
| 8 | RBF-SVM: size of the training subsample on Fashion-MNIST (and Adult?) | | |
| 9 | Hyperparameter search space and budget for each method; validation metric used for selection | | |
| **Code** | | | |
| 10 | Public interface of the MLP (class/function names, arguments, return values) | | |
| 11 | Interface of the dataset loaders (what they return) | | |
| 12 | Result file format and where results are stored | | |
| 13 | Python version; pinned dependency versions? | | |
| 14 | Test framework (e.g. pytest)? Continuous integration on GitHub? | | |
| 15 | Packaging / import path for `src/` (e.g. `pip install -e .` vs. `PYTHONPATH`) | | |
| **Repository and writing** | | | |
| 16 | Default branch name; branch / pull-request / review rules | | |
| 17 | License for the code | | |
| 18 | Report: one `main.tex` or one file per section? Overleaf or the repo as the source of truth? | | |
| 19 | Shared notation for the report (symbols, vector/matrix conventions) | | |
