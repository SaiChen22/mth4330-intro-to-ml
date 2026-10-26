"""
- [ ] random forest
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest(task, max_depth, random_state=0):
    """
    task: distinguishes classification and regression.
    max_depth: tuning parameter
    random_state: seeds 0–4
    """
    if task == "classification":
        return RandomForestClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
    elif task == "regression":
        return RandomForestRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown task: {task}")