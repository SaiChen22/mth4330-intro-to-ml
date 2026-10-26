"""
- [ ] RBF-kernel SVM / SVR (subsampled training set on Fashion-MNIST if needed)
"""

from sklearn.svm import SVC, SVR


def svm(task, C, random_state=0, probability=False):
    """
    task: distinguishes classification and regression.
    C: parameter
    random_state: seeds 0–4
    probability: SVM probabilities disabled during tuning, then set probability=True for the final fit.
    """
    if task == "classification":
        return SVC(
            C=C,
            kernel="rbf",
            random_state=random_state,
            probability=probability
        )
    elif task == "regression":
        return SVR(
            C=C,
            kernel="rbf"
        )
    else:
        raise ValueError(f"Unknown task: {task}")