"""Loss functions and their gradients.

Scope (approval):
- [x] softmax cross-entropy (classification)
- [x] squared error (regression)
- [x] L2 weight-decay term (lambda/2) sum_l ||W^(l)||_F^2

Data losses return ``(loss, grad)`` where ``loss`` is the mean over the batch and ``grad`` is
d loss / d (network output), already divided by the batch size n. The L2 term is a plain sum
over the weights (biases are not penalised) and is NOT rescaled by the batch size.
"""

import numpy as np


def softmax(logits):
    """Row-wise softmax; subtracting the row max keeps exp() from overflowing."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy of ``softmax(logits)`` (shape (n, k)) against integer labels y (n,).

    Returns ``(loss, grad)`` with grad = (softmax(logits) - onehot(y)) / n, shape (n, k).
    """
    n = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()
    grad = np.exp(log_probs)
    grad[np.arange(n), y] -= 1
    return loss, grad / n


def squared_error(y_hat, y):
    """Mean of 1/2 (y_hat - y)^2 for predictions and targets of shape (n,).

    Returns ``(loss, grad)`` with grad = (y_hat - y) / n.
    """
    diff = y_hat - y
    return 0.5 * np.mean(diff**2), diff / diff.shape[0]


def l2_penalty(weights, lam):
    """(lam / 2) * sum of ||W||^2 over the list of weight matrices (biases excluded)."""
    return 0.5 * lam * sum(np.sum(W**2) for W in weights)


def l2_grad(W, lam):
    """Gradient of the L2 penalty with respect to one weight matrix: lam * W."""
    return lam * W
