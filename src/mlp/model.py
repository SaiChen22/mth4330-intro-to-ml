"""The MLP model: training and prediction.

Scope (approval):
- [x] mini-batch gradient training
- [x] L2 weight decay, dropout, early stopping
- [x] classification (softmax output) and regression (linear output)
- [x] prediction

Interface: docs/open-questions.md #10 (``record_norms`` is deferred).

Structure: Dense -> activation -> Dropout for each hidden layer, then a final Dense. The output is
k logits for classification (softmax inside the loss) and one linear output for regression.
"""

import numpy as np

from . import activations, init, losses, optim
from .layers import Dense, Dropout

_TASKS = ("classification", "regression")
_OPTIMIZERS = ("sgd", "momentum", "adam")


class MLP:
    def __init__(
        self,
        *,
        hidden_sizes,
        activation,
        task,
        init,
        weight_decay=0.0,
        dropout=0.0,
        optimizer,
        lr,
        momentum=0.0,
        batch_size,
        epochs,
        patience,
        seed,
        dtype=np.float64,
    ):
        if task not in _TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {_TASKS}")
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {_OPTIMIZERS}")
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.task = task
        self.init = init
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.dtype = dtype

    def _build(self, n_in, n_out):
        """Create the layers for ``n_in`` inputs and ``n_out`` outputs (k classes, or 1)."""
        init_fn = init.get(self.init)
        self._act, self._act_grad = activations.get(self.activation)
        self.rng_ = np.random.default_rng(self.seed)
        sizes = (n_in, *self.hidden_sizes, n_out)
        self.layers_ = [
            Dense(sizes[i], sizes[i + 1], init_fn, self.rng_, self.dtype) for i in range(len(sizes) - 1)
        ]
        self._dropouts = [Dropout(self.dropout, self.rng_) for _ in self.hidden_sizes]
        return self

    def _forward(self, X, training):
        """Return the network output: logits (n, k), or predictions (n,) for regression."""
        h = X
        self._z = []
        for dense, drop in zip(self.layers_[:-1], self._dropouts):
            z = dense.forward(h)
            self._z.append(z)
            h = drop.forward(self._act(z), training)
        out = self.layers_[-1].forward(h)
        return out[:, 0] if self.task == "regression" else out

    def _backward(self, grad_out):
        """Backpropagate d loss / d output; leaves dW, db in every Dense layer."""
        g = grad_out[:, None] if self.task == "regression" else grad_out
        g = self.layers_[-1].backward(g)
        for dense, drop, z in zip(self.layers_[-2::-1], self._dropouts[::-1], self._z[::-1]):
            g = drop.backward(g) * self._act_grad(z)
            g = dense.backward(g)

    def _loss_and_grads(self, X, y, training=True):
        """Mean data loss + L2 penalty on (X, y); gradients are left in the layers (L2 included)."""
        out = self._forward(X, training)
        loss_fn = losses.softmax_cross_entropy if self.task == "classification" else losses.squared_error
        data_loss, grad_out = loss_fn(out, y)
        self._backward(grad_out)
        if self.weight_decay:
            for dense in self.layers_:
                dense.dW += losses.l2_grad(dense.W, self.weight_decay)
        return data_loss + losses.l2_penalty([d.W for d in self.layers_], self.weight_decay)

    def _data_loss(self, X, y):
        """Mean data loss (no L2 term) in prediction mode (no dropout)."""
        loss_fn = losses.softmax_cross_entropy if self.task == "classification" else losses.squared_error
        return loss_fn(self._forward(X, training=False), y)[0]

    def _params(self):
        return [p for d in self.layers_ for p in (d.W, d.b)]

    def fit(self, X, y, X_val=None, y_val=None):
        """Train with mini-batches; early stopping on the validation data loss if it is given."""
        X = np.asarray(X, dtype=self.dtype)
        if self.task == "classification":
            y = np.asarray(y, dtype=int)
            n_out = int(y.max()) + 1
        else:
            y = np.asarray(y, dtype=self.dtype)
            n_out = 1
        validate = X_val is not None
        if validate:
            X_val = np.asarray(X_val, dtype=self.dtype)
            y_val = np.asarray(y_val, dtype=int if self.task == "classification" else self.dtype)

        self._build(X.shape[1], n_out)
        opt = optim.get(self.optimizer, self.lr, self.momentum)
        n = X.shape[0]
        self.history_ = {"train": [], "val": []}
        best_loss, best_params, wait = np.inf, None, 0

        for _ in range(self.epochs):
            order = self.rng_.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                self._loss_and_grads(X[idx], y[idx], training=True)
                grads = [g for d in self.layers_ for g in (d.dW, d.db)]
                opt.step(self._params(), grads)
            self.history_["train"].append(self._data_loss(X, y))
            if not validate:
                continue
            val_loss = self._data_loss(X_val, y_val)
            self.history_["val"].append(val_loss)
            if val_loss < best_loss:
                best_loss, best_params, wait = val_loss, [p.copy() for p in self._params()], 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_params is not None:
            for p, best in zip(self._params(), best_params):
                p[...] = best
        return self

    def predict(self, X):
        """Class labels (classification) or predicted values (regression)."""
        out = self._forward(np.asarray(X, dtype=self.dtype), training=False)
        return out.argmax(axis=1) if self.task == "classification" else out

    def predict_proba(self, X):
        """Class probabilities, shape (n, k); classification only."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for task='classification'")
        return losses.softmax(self._forward(np.asarray(X, dtype=self.dtype), training=False))

    def n_parameters(self):
        return sum(d.W.size + d.b.size for d in self.layers_)
