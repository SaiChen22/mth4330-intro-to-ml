"""Our multilayer perceptron, written by the team.

Rules (from the approved proposal and the course guidelines):
- NumPy only; no automatic differentiation (no PyTorch, JAX, autograd).
- Layers, forward pass, backpropagation, losses, optimisers, initialisation and
  regularisation are all implemented here.

Model: h^(0) = x,  h^(l) = sigma(W^(l) h^(l-1) + b^(l)),  f_theta(x) = W^(L) h^(L-1) + b^(L).
Objective: (1/n) sum_i L(f_theta(x_i), y_i) + (lambda/2) sum_l ||W^(l)||_F^2.
With zero hidden layers this is softmax/logistic regression (classification) or
ridge regression (regression).

Must support: multiclass classification (Fashion-MNIST), binary classification (Adult),
regression (California Housing).
"""
