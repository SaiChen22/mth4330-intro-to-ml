"""With no hidden layer the MLP is ridge / softmax regression: it matches scikit-learn.

Our loss is mean data loss + (lambda/2) ||W||^2 (biases unpenalised, docs/open-questions.md #10), so
- ridge:               Ridge(alpha = n * lambda)
- softmax, K >= 3:     LogisticRegression(C = 1 / (n * lambda))
- two-output softmax:  LogisticRegression(C = 2 / (n * lambda)); the penalty forces w1 = -w2, which
                       halves the penalty on the logit difference, hence the factor 2.
Full-batch momentum on the convex problem is run to convergence; scikit-learn is solved to tight tolerance.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from mlp.model import MLP

N, D, LAM = 500, 8, 0.01


def fit(task, X, y):
    return MLP(
        hidden_sizes=(), activation="relu", task=task, init="he", weight_decay=LAM,
        optimizer="momentum", lr=0.5, momentum=0.9, batch_size=N, epochs=3000, patience=1, seed=0,
    ).fit(X, y)


def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D))
    return rng, X, rng.normal(size=D)


def test_ridge():
    rng, X, w = data()
    y = X @ w + 0.3 * rng.normal(size=N)
    ours = fit("regression", X, y)
    ref = Ridge(alpha=N * LAM, solver="cholesky").fit(X, y)
    assert np.allclose(ours.layers_[0].W[0], ref.coef_, atol=1e-8)
    assert np.isclose(ours.layers_[0].b[0], ref.intercept_, atol=1e-8)


def test_binary_uses_c_twice_as_large():
    rng, X, w = data()
    y = (X @ w + rng.logistic(size=N) > 0).astype(int)
    ours = fit("classification", X, y).predict_proba(X)[:, 1]
    right = LogisticRegression(C=2 / (N * LAM), tol=1e-12, max_iter=10000).fit(X, y)
    wrong = LogisticRegression(C=1 / (N * LAM), tol=1e-12, max_iter=10000).fit(X, y)
    assert np.abs(ours - right.predict_proba(X)[:, 1]).max() < 1e-6
    assert np.abs(ours - wrong.predict_proba(X)[:, 1]).max() > 1e-3  # C = 1 / (n lambda) is off


def test_three_class_softmax():
    rng, X, _ = data()
    y = np.argmax(X @ rng.normal(size=(D, 3)) + rng.gumbel(size=(N, 3)), axis=1)
    ours = fit("classification", X, y).predict_proba(X)
    ref = LogisticRegression(C=1 / (N * LAM), tol=1e-12, max_iter=10000).fit(X, y)
    assert np.abs(ours - ref.predict_proba(X)).max() < 1e-6
