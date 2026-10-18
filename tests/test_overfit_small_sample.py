"""The MLP can overfit a very small training sample (approval; guidelines: failure analysis).
"""

import numpy as np

from mlp.model import MLP


def test_memorises_random_labels():
    # 20 points with random labels: only memorisation can reach 100% training accuracy
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10))
    y = rng.integers(0, 3, size=20)
    model = MLP(
        hidden_sizes=(64,), activation="relu", task="classification", init="he",
        optimizer="adam", lr=0.01, batch_size=20, epochs=500, patience=1, seed=0,
    ).fit(X, y)
    assert np.array_equal(model.predict(X), y)
