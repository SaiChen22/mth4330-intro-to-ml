"""Linear baselines (scikit-learn): ridge regression, the no-hidden-layer MLP for regression.

Correspondence with our MLP loss (1/2 mean squared error + (lambda/2) ||W||^2, biases unpenalised):
Ridge(alpha = n * lambda), where n is the number of training examples (docs/open-questions.md #10).
"""

from sklearn.linear_model import Ridge


def ridge(alpha):
    return Ridge(alpha=alpha)
