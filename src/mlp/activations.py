"""Activation functions and their derivatives.

Scope (approval):
- [x] ReLU
- [x] tanh
- [x] sigmoid

Each ``f(z)`` has a matching ``f_grad(z)``: the derivative sigma'(z) with respect to the
pre-activation z, evaluated elementwise. ``get(name)`` returns the pair for a name in
{"relu", "tanh", "sigmoid"}.

Needed later for the gradient-flow experiment: the fraction of dead ReLUs per layer.
"""

import numpy as np


def relu(z):
    return np.maximum(z, 0)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


def sigmoid(z):
    # exp of a non-positive argument only, so large |z| cannot overflow
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


_ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


def get(name):
    """Return ``(f, f_grad)`` for ``name`` in {"relu", "tanh", "sigmoid"}."""
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}")
