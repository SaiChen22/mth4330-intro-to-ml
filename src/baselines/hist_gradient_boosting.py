"""
- [ ] histogram gradient boosting
"""

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def hist_gradient_boosting( task, learning_rate, random_state=0):
    """
    task: distinguishes classification and regression.
    learning_rate: tuning parameter
    random_state: seeds 0–4
    """
    if task == "classification":
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            random_state=random_state
        )
    elif task == "regression":
        return HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown task: {task}")