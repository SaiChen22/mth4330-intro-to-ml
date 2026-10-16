# Experiments

Planned experiments, taken from the approved proposal and the project guidelines.
Owners, commands and settings are filled in once decided (see `docs/open-questions.md`).

| # | Experiment | What is measured | Source | Owner | Status |
|---|---|---|---|---|---|
| 1 | Three-dataset comparison: our MLP vs. every baseline | dataset metrics, training time, parameter count; mean ± s.d. over 5 seeds | approval §4–5 | | not started |
| 2 | Initialisation and gradient flow: Xavier vs. He scaling | per-layer activation and gradient-norm measurements; vanishing / exploding gradients; dead ReLUs | approval §2 | | not started |
| 3 | Depth separation with composed tent maps tᵏ | deep vs. shallow networks tested empirically at fixed parameter count | approval §2 | | not started |
| 4 | Optimisation: SGD, momentum, Adam | behaviour of each optimiser; learning-rate sensitivity | approval §2 | | not started |
| 5 | Sensitivity to important choices | results across a sensible range of hyperparameters | guidelines (final report §6, failure analysis) | | not started |
| 6 | Failure analysis | error patterns across datasets and baselines (e.g. Fashion-MNIST confusion matrix) | guidelines (failure analysis) | | not started |

Correctness checks (gradient check, PyTorch agreement, XOR, tiny-sample overfit) live in `tests/`.
