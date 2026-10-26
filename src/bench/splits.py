"""Fixed train / validation / test splits.

Protocol (approval):
- identical fixed splits for every method
- Fashion-MNIST: official test set
- Adult, California Housing: stratified 70/15/15

Decided (docs/open-questions.md): #6 split seed 4330; #3 Fashion-MNIST validation is a
stratified 10% of the official training set; #5 California Housing stratifies on 10 quantile bins
of the target (the loader makes the bins).
"""

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_70_15_15(labels, seed):
    """Return index arrays ``(train, val, test)`` for a stratified 70/15/15 split of ``labels``."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=seed)
    return train, val, test


def stratified_holdout(labels, frac, seed):
    """Return index arrays ``(rest, holdout)``: a stratified ``frac`` of ``labels`` is held out."""
    labels = np.asarray(labels)
    return train_test_split(np.arange(len(labels)), test_size=frac, stratify=labels, random_state=seed)
