from __future__ import annotations

from typing import Optional

import torch


def cofactor3_torch(M: torch.Tensor) -> torch.Tensor:
    C = torch.empty_like(M)
    A = M
    C[..., 0, 0] = A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]
    C[..., 0, 1] = A[..., 1, 2] * A[..., 2, 0] - A[..., 1, 0] * A[..., 2, 2]
    C[..., 0, 2] = A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0]
    C[..., 1, 0] = A[..., 0, 2] * A[..., 2, 1] - A[..., 0, 1] * A[..., 2, 2]
    C[..., 1, 1] = A[..., 0, 0] * A[..., 2, 2] - A[..., 0, 2] * A[..., 2, 0]
    C[..., 1, 2] = A[..., 0, 1] * A[..., 2, 0] - A[..., 0, 0] * A[..., 2, 1]
    C[..., 2, 0] = A[..., 0, 1] * A[..., 1, 2] - A[..., 0, 2] * A[..., 1, 1]
    C[..., 2, 1] = A[..., 0, 2] * A[..., 1, 0] - A[..., 0, 0] * A[..., 1, 2]
    C[..., 2, 2] = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    return C


def deep_coro_forward_torch(M: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    X = M
    for aa, bb in zip(alpha, beta):
        Y = aa * X + bb * cofactor3_torch(X)
        det = torch.linalg.det(Y)
        sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
        rho = sign * torch.clamp(det.abs(), min=eps).pow(-1.0 / 3.0)
        X = rho[..., None, None] * Y
    return X


class DeepCOROModule(torch.nn.Module):
    """Trainable or fixed Deep CORO head for PyTorch training.

    The OpenCL operator is forward-only and not used here.  This module is a
    differentiable reference suitable for training the CORO coefficients or a
    front-end network. Export the trained alpha/beta arrays to the OpenCL runtime.
    """

    def __init__(self, layers: int = 8, trainable: bool = True, init_alpha: float = 1.0, init_beta: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)
        alpha = torch.full((layers,), float(init_alpha), dtype=torch.float32)
        beta = torch.full((layers,), float(init_beta), dtype=torch.float32)
        if trainable:
            self.alpha = torch.nn.Parameter(alpha)
            self.beta = torch.nn.Parameter(beta)
        else:
            self.register_buffer("alpha", alpha)
            self.register_buffer("beta", beta)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        return deep_coro_forward_torch(M, self.alpha.to(M.dtype), self.beta.to(M.dtype), self.eps)
