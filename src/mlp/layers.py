"""Network layers: forward and backward passes.

Scope (approval):
- [x] fully connected (affine) layer: ``Dense``
- [x] dropout: ``Dropout``

Backpropagation (matrix form, approval):
  delta^(l) = (W^(l+1)^T delta^(l+1)) * sigma'(z^(l))
  grad_{W^(l)} L = delta^(l) h^(l-1)^T
Cost O(sum_l n_(l-1) n_l) per example.

Data is stored one example per row, so a batch H of shape (n, fan_in) gives Z = H W^T + b and,
with G = dL/dZ of shape (n, fan_out), dW = G^T H, db = sum of the rows of G, dH = G W.
The nonlinearity is applied by the model between layers (see ``mlp.activations``). The L2 gradient
is added by the model, not here.
"""

import numpy as np


class Dense:
    """Affine layer z = W h + b with W of shape (fan_out, fan_in)."""

    def __init__(self, fan_in, fan_out, init_fn, rng, dtype=np.float64):
        self.W = init_fn(fan_in, fan_out, rng, dtype)
        self.b = np.zeros(fan_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, H):
        self._H = H
        return H @ self.W.T + self.b

    def backward(self, G):
        """Given G = dL/dZ, store dW, db and return dL/dH."""
        self.dW = G.T @ self._H
        self.db = G.sum(axis=0)
        return G @ self.W


class Dropout:
    """Inverted dropout: drop each unit with probability p and scale the rest by 1 / (1 - p)."""

    def __init__(self, p, rng):
        self.p = p
        self.rng = rng

    def forward(self, H, training=True):
        if not training or self.p == 0:
            self._mask = None
            return H
        keep = self.rng.random(H.shape, dtype=H.dtype) >= self.p
        self._mask = keep.astype(H.dtype) / (1 - self.p)
        return H * self._mask

    def backward(self, G):
        return G if self._mask is None else G * self._mask
