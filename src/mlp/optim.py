"""Optimisers.

Scope (approval):
- [x] SGD (mini-batch)
- [x] momentum
- [x] Adam

Each optimiser has ``step(params, grads)``: ``params`` and ``grads`` are matching lists of arrays and
the parameters are updated in place. ``get(name, lr, momentum)`` builds one by name.

- SGD:      p <- p - lr g
- momentum: v <- mu v + g;  p <- p - lr v
- Adam:     m <- b1 m + (1-b1) g;  v <- b2 v + (1-b2) g^2;
            p <- p - lr (m / (1-b1^t)) / (sqrt(v / (1-b2^t)) + eps),  b1 = 0.9, b2 = 0.999, eps = 1e-8
"""

import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.mu = momentum
        self._v = None

    def step(self, params, grads):
        if self._v is None:
            self._v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self._v):
            v *= self.mu
            v += g
            p -= self.lr * v


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self._m = self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        c1 = 1 - self.b1**self._t
        c2 = 1 - self.b2**self._t
        for p, g, m, v in zip(params, grads, self._m, self._v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def get(name, lr, momentum=0.0):
    """Return an optimiser for ``name`` in {"sgd", "momentum", "adam"} (``momentum`` is only used by "momentum")."""
    if name == "sgd":
        return SGD(lr)
    if name == "momentum":
        return Momentum(lr, momentum)
    if name == "adam":
        return Adam(lr)
    raise ValueError(f"unknown optimizer {name!r}; expected one of ['adam', 'momentum', 'sgd']")
