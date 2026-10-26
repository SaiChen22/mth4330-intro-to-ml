"""
I am not sure if you just want logistic regression alone or with ridge in linear 
- [ ] softmax / logistic regression (classification) and ridge regression (regression)
      = the no-hidden-layer MLP
"""
from sklearn.linear_model import Ridge, LogisticRegression

def ridge(alpha):
    return Ridge(alpha=alpha)


def logistic(C, random_state=0):
    return LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state
    )
