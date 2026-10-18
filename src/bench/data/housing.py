"""California Housing (StatLib / scikit-learn).

Facts (approval): 20,640 block groups, 8 numeric features; regression of median house value.
Split (approval): stratified 70/15/15.
Preprocessing (approval): scaling, fit on the training split only.

Decided (docs/open-questions.md #5): stratify on 10 quantile bins of the target; standardise the
8 features (fit on train only); the target is left unscaled (units of $100k).
Loader interface (#11): ``load_housing(split_seed=4330)`` returns
``X_train, y_train, X_val, y_val, X_test, y_test`` as NumPy arrays.
The raw file is downloaded by scikit-learn into data/ on first use.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from ..splits import stratified_70_15_15

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_housing(split_seed=4330):
    X, y = fetch_california_housing(data_home=_DATA_DIR, return_X_y=True)
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])  # 9 inner edges -> 10 bins
    train, val, test = stratified_70_15_15(np.digitize(y, edges), split_seed)
    scaler = StandardScaler().fit(X[train])
    return (
        scaler.transform(X[train]), y[train],
        scaler.transform(X[val]), y[val],
        scaler.transform(X[test]), y[test],
    )
