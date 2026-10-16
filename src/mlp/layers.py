"""Network layers: forward and backward passes.

Scope (approval):
- [ ] fully connected (affine) layer
- [ ] dropout

Backpropagation (matrix form, approval):
  delta^(l) = (W^(l+1)^T delta^(l+1)) * sigma'(z^(l))
  grad_{W^(l)} L = delta^(l) h^(l-1)^T
Cost O(sum_l n_(l-1) n_l) per example.
"""
