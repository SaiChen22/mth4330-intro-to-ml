"""Fashion-MNIST (Zalando; OpenML).

Facts (approval): 70,000 grayscale 28x28 images, 10 classes; multiclass image classification.
Split (approval): official test set.

Decided (docs/open-questions.md): #3 validation set = stratified 10% of the official 60k training
set (54k train / 6k validation); #7 pixels divided by 255 and flattened to 784 features, nothing else.
Loader: ``load_fmnist(split_seed=4330)``; X is float32 in [0, 1], y holds class indices 0-9.
Source: the official IDX files from Zalando's repository (this fixes the official train/test split).
"""

import gzip

import numpy as np

from . import data_path
from ..splits import stratified_holdout


def _read(filename, offset):
    with gzip.open(data_path(filename)) as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def _images(filename):
    return _read(filename, 16).reshape(-1, 784).astype(np.float32) / 255


def load_fmnist(split_seed=4330):
    X, y = _images("train-images-idx3-ubyte.gz"), _read("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
    X_test, y_test = _images("t10k-images-idx3-ubyte.gz"), _read("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)
    train, val = stratified_holdout(y, 0.10, split_seed)
    return X[train], y[train], X[val], y[val], X_test, y_test
