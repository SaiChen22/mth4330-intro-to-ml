# Multilayer Perceptrons: Mathematical Foundations, a From-Scratch Implementation, and a Three-Dataset Study

MTH 4330 (Fall 2026) final project — **Leo Chen, Sai Kaung San, Priyanshu Rana**

> **Status: skeleton.** Directory layout and scope only — no implementation yet.
> Items not yet decided by the team are listed in [`docs/open-questions.md`](docs/open-questions.md).

## Project

We study a fully connected feedforward neural network (multilayer perceptron, MLP) trained by
backpropagation and mini-batch gradient methods, implement it ourselves, and evaluate it on three
datasets against baseline methods.

**Algorithm scope (from the approved proposal)**
- Activations: ReLU, tanh, sigmoid
- Losses: softmax cross-entropy (classification), squared error (regression)
- Regularisation: L2 weight decay, dropout, early stopping
- Optimisers: SGD, momentum, Adam
- Initialisation: Xavier / He

**Mathematical scope**
- Model and regularised objective; the zero-hidden-layer case is softmax/logistic regression
  (classification) or ridge regression (regression); L2 as a Gaussian MAP prior
- Matrix-form backpropagation and its cost
- Non-convexity; behaviour of SGD, momentum and Adam; learning-rate sensitivity
- Variance propagation, Xavier/He scaling, vanishing/exploding gradients, dead ReLUs
- Universal approximation (statement) and depth separation with composed tent maps

## Datasets

| Dataset | Source | Size | Task |
|---|---|---|---|
| Fashion-MNIST | Zalando; OpenML | 70,000 grayscale 28×28 images, 10 classes | multiclass image classification |
| Adult Census Income | UCI | 48,842 records, 14 mixed numeric/categorical features | binary classification of income > $50K (≈ 24 % positive) |
| California Housing | StatLib / scikit-learn | 20,640 block groups, 8 numeric features | regression of median house value |

Data is not committed to this repository. Download the raw files into `data/` with

```bash
python scripts/download_data.py
```

The loaders in `src/bench/data/` read them from `data/` and do all preprocessing (fit on the training split only).

## Baselines

All from scikit-learn: softmax/logistic regression (classification) and ridge regression (regression),
i.e. the no-hidden-layer MLP; random forest; histogram gradient boosting; RBF-kernel SVM/SVR
(subsampled training set on Fashion-MNIST if needed).

PyTorch and scikit-learn's `MLPClassifier`/`MLPRegressor` are used **only** as correctness references,
kept in a separate directory (`src/reference/`).

## Evaluation protocol

| Dataset | Metrics |
|---|---|
| Fashion-MNIST | accuracy, macro-F1, confusion matrix |
| Adult | ROC-AUC, F1 (positive class), balanced accuracy, log-loss |
| California Housing | RMSE, MAE, R² |
| All | training time, parameter count |

- Identical fixed train/validation/test splits for every method: official test set for Fashion-MNIST;
  stratified 70/15/15 otherwise.
- Preprocessing (scaling, one-hot encoding) fit on training data only.
- Hyperparameters tuned on the validation set with comparable budgets.
- Results reported as mean ± s.d. over 5 random seeds.

## Correctness checks

Finite-difference gradient checking; agreement with PyTorch autograd on identical weights;
XOR; overfitting a tiny training sample.

## Implementation rules

- Python 3. The MLP (layers, forward pass, backpropagation, losses, optimisers, initialisation,
  regularisation) is written by the team in **NumPy**, with **no automatic differentiation**.
- pandas and scikit-learn are used for data handling, preprocessing and baselines; matplotlib for figures.
- Library implementations (baselines, references) are kept separate from our own code.

## Repository structure

```
.
├── README.md
├── requirements.txt              core dependencies
├── requirements-reference.txt    PyTorch, for correctness checks only
├── .gitignore
├── docs/
│   └── open-questions.md         decisions still to be made by the team
├── src/
│   ├── mlp/                      our MLP (NumPy only)
│   │   ├── activations.py        ReLU, tanh, sigmoid
│   │   ├── losses.py             softmax cross-entropy, squared error, L2 penalty
│   │   ├── init.py               Xavier / He initialisation
│   │   ├── layers.py             fully connected layer, dropout
│   │   ├── optim.py              SGD, momentum, Adam
│   │   └── model.py              training (mini-batches, early stopping) and prediction
│   ├── bench/                    shared by every method
│   │   ├── data/                 fmnist.py, adult.py, housing.py
│   │   ├── splits.py             fixed train/validation/test splits
│   │   ├── metrics.py            per-dataset metrics
│   │   ├── runner.py             run a method over seeds
│   │   ├── results.py            store and summarise results
│   │   └── plotting.py           figures
│   ├── baselines/                scikit-learn baselines
│   └── reference/                PyTorch / scikit-learn MLP, correctness checks only
├── tests/                        gradient check, PyTorch agreement, XOR, tiny-sample overfit
├── scripts/                      download_data.py, reproduce_results.py
├── experiments/README.md         planned experiments
├── data/                         downloaded datasets (not committed)
├── results/                      experiment outputs
└── report/                       LaTeX report (course ICML template)
```

## Setup

```bash
pip install -e .                            # installs requirements.txt; makes mlp, bench, baselines, reference importable
pip install -r requirements-reference.txt   # only for the PyTorch correctness check
```

Tests: `pytest`.

## Reproducing results

```bash
python scripts/download_data.py
python scripts/reproduce_results.py      # writes results/<dataset>_<method>.json
python scripts/summarize_results.py      # prints the tables from results/*.json
```

Currently the script runs only the California Housing pilot (ridge vs. our MLP).

## Milestones

| Item | Due |
|---|---|
| Project approval | September 18 (submitted) |
| Preliminary report (≤ 4 pages) | October 21 |
| Presentation (20–30 min) | final weeks of class |
| Final report (5–6 pages) | final exam day |

## References

1. D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning representations by back-propagating errors. *Nature*, 323:533–536, 1986.
2. X. Glorot, Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 2010.
3. M. Telgarsky. Benefits of depth in neural networks. *COLT*, 2016.
