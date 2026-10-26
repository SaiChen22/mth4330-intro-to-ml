"""Adult Census Income (UCI).

Facts (approval): 48,842 records, 14 mixed numeric/categorical features;
binary classification of income > $50K, imbalanced (~24% positive).
Split (approval): stratified 70/15/15.
Preprocessing (approval): scaling, one-hot encoding, fit on the training split only.

Decided (docs/open-questions.md #4): pool adult.data and adult.test (skip the first line of
adult.test and strip the trailing "." from its labels; strip leading spaces); stratify on the
label; ``?`` (only in categorical columns) becomes its own "Unknown" category; keep all 14
features; standardise the numeric columns and one-hot encode the categorical ones (unseen
categories ignored), both fit on train only.
Loader: ``load_adult(split_seed=4330)``; y is 1 for income > $50K, else 0.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data_path
from ..splits import stratified_70_15_15

_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
_COLUMNS = (
    ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation"]
    + ["relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"]
    + ["native-country", "income"]
)


def _read(filename, **kwargs):
    return pd.read_csv(data_path(filename), names=_COLUMNS, skipinitialspace=True, **kwargs)


def load_adult(split_seed=4330):
    df = pd.concat([_read("adult.data"), _read("adult.test", skiprows=1)], ignore_index=True)
    y = (df.pop("income").str.rstrip(".") == ">50K").astype(int).to_numpy()
    df[_CATEGORICAL] = df[_CATEGORICAL].replace("?", "Unknown")
    train, val, test = stratified_70_15_15(y, split_seed)
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), _NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL),
        ]
    ).fit(df.iloc[train])
    return (
        prep.transform(df.iloc[train]), y[train],
        prep.transform(df.iloc[val]), y[val],
        prep.transform(df.iloc[test]), y[test],
    )
