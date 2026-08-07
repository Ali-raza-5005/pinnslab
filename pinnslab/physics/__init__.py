"""PDE residual operators and diffops.

Residuals return per-point tensors of shape (N,), never scalars (DESIGN.md §4).
The residuals themselves live with the benchmark or paper that defines them;
what is generic — and therefore here — is how to differentiate.
"""

from pinnslab.physics.diffops import (
    gradient,
    laplacian,
    partial,
    requires_grad,
    second_partial,
)

__all__ = [
    "gradient",
    "laplacian",
    "partial",
    "requires_grad",
    "second_partial",
]
