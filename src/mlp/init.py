"""Weight initialisation.

Scope (approval):
- [x] Xavier initialisation
- [x] He initialisation

Both are derived in the report from variance propagation through layers.

Each initialiser returns a weight matrix W of shape (fan_out, fan_in), matching the report's
z = W h + b, with Var(W_ij) = 2 / (fan_in + fan_out) (Xavier) or 2 / fan_in (He).
Biases are initialised to zero by the layer, not here. ``get(name)`` looks one up by name.
"""

import numpy as np


def xavier(fan_in, fan_out, rng, dtype=np.float64):
    """Xavier / Glorot uniform: W_ij ~ U(-a, a), a = sqrt(6 / (fan_in + fan_out))."""
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_out, fan_in)).astype(dtype)


def he(fan_in, fan_out, rng, dtype=np.float64):
    """He normal: W_ij ~ N(0, 2 / fan_in)."""
    return (rng.standard_normal((fan_out, fan_in)) * np.sqrt(2.0 / fan_in)).astype(dtype)


_INITS = {"xavier": xavier, "he": he}


def get(name):
    """Return the initialiser for ``name`` in {"xavier", "he"}."""
    try:
        return _INITS[name]
    except KeyError:
        raise ValueError(f"unknown init {name!r}; expected one of {sorted(_INITS)}")
