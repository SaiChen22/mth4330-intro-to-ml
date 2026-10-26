"""The MLP learns XOR (approval: correctness checks).
"""

import numpy as np

from mlp.model import MLP


def test_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = MLP(
        hidden_sizes=(8,), activation="tanh", task="classification", init="xavier",
        optimizer="adam", lr=0.05, batch_size=4, epochs=500, patience=1, seed=0,
    ).fit(X, y)
    assert np.array_equal(model.predict(X), y)
