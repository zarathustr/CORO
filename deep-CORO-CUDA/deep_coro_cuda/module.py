"""Trainable Deep CORO torch module."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .ops import deep_coro_forward, deep_coro_forward_torch


class DeepCOROModule(nn.Module):
    """A trainable or fixed stack of CORO projection layers.

    Parameters
    ----------
    layers:
        Number of CORO layers.
    trainable:
        If true, `alpha` and `beta` are `nn.Parameter`s.
    alpha0, beta0:
        Initialization.  `alpha0 + beta0 = 2` is the usual stable choice.
    use_extension:
        If true, inference with no gradients will use the C++/CUDA extension
        when installed.  Training always uses the PyTorch path.
    """

    def __init__(
        self,
        layers: int = 8,
        trainable: bool = False,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        eps: float = 1e-12,
        use_extension: bool = True,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        a = torch.full((layers,), float(alpha0), dtype=torch.float32)
        b = torch.full((layers,), float(beta0), dtype=torch.float32)
        if trainable:
            self.alpha = nn.Parameter(a)
            self.beta = nn.Parameter(b)
        else:
            self.register_buffer("alpha", a)
            self.register_buffer("beta", b)
        self.layers = layers
        self.eps = eps
        self.use_extension = use_extension

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(device=M.device, dtype=M.dtype)
        beta = self.beta.to(device=M.device, dtype=M.dtype)
        if self.training or M.requires_grad:
            return deep_coro_forward_torch(M, alpha, beta, self.eps)
        return deep_coro_forward(M, alpha, beta, self.eps, use_extension=self.use_extension)
