"""Softmax / logistic regression baseline (scikit-learn): the no-hidden-layer MLP for classification.

Ridge regression, the regression counterpart, is in linear.py.
Correspondence with our MLP loss (cross-entropy + (lambda/2) ||W||^2, biases unpenalised):
C = 1 / (n * lambda) for K >= 3 classes and C = 2 / (n * lambda) for the two-output softmax used on
Adult, where n is the number of training examples (docs/open-questions.md #10).
"""

from sklearn.linear_model import LogisticRegression


def logistic(C, random_state=0):
    """
    C: inverse regularisation strength (tuning parameter)
    random_state: seeds 0-4
    """
    return LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state
    )
