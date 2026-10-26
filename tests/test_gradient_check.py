"""Finite-difference gradient checking of our backpropagation (approval: correctness checks).
"""

import numpy as np
import pytest

from mlp.layers import Dropout
from mlp.model import MLP


def make(task, activation, hidden_sizes, weight_decay):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 5))
    if task == "classification":
        y, n_out = rng.integers(0, 3, size=8), 3
    else:
        y, n_out = rng.normal(size=8), 1
    model = MLP(
        hidden_sizes=hidden_sizes, activation=activation, task=task, init="he",
        weight_decay=weight_decay, dropout=0.0, optimizer="sgd", lr=0.1,
        batch_size=8, epochs=1, patience=1, seed=0,
    )
    return model._build(5, n_out), X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
@pytest.mark.parametrize("hidden_sizes", [(), (6, 4)])
def test_backprop_matches_finite_differences(task, activation, hidden_sizes):
    model, X, y = make(task, activation, hidden_sizes, weight_decay=0.1)
    model._loss_and_grads(X, y)
    h = 1e-6
    for layer in model.layers_:
        for param, grad in ((layer.W, layer.dW), (layer.b, layer.db)):
            num = np.zeros_like(param)
            for i in np.ndindex(*param.shape):
                old = param[i]
                param[i] = old + h
                up = model._loss_and_grads(X, y)
                param[i] = old - h
                down = model._loss_and_grads(X, y)
                param[i] = old
                num[i] = (up - down) / (2 * h)
            model._loss_and_grads(X, y)  # restore the analytic gradient overwritten above
            assert np.allclose(grad, num, rtol=1e-5, atol=1e-8)


def test_dropout_backward_uses_forward_mask():
    drop = Dropout(0.5, np.random.default_rng(0))
    H = np.ones((100, 20))
    out = drop.forward(H, training=True)
    assert set(np.unique(out)) <= {0.0, 2.0}  # dropped, or kept and scaled by 1 / (1 - p)
    assert np.array_equal(drop.backward(H), out)
    assert drop.forward(H, training=False) is H  # identity at prediction time
