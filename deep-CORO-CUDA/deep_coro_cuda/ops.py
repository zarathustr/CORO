"""Python wrapper and differentiable PyTorch fallback for Deep CORO."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:  # installed after `python -m pip install -e .`
    import deep_coro_ext  # type: ignore
except Exception:  # pragma: no cover - expected in source-only mode
    deep_coro_ext = None


def cofactor3(M: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 cofactor matrix for each matrix in a batch.

    Parameters
    ----------
    M:
        Tensor with shape `(..., 3, 3)`.
    """
    if M.shape[-2:] != (3, 3):
        raise ValueError("cofactor3 expects shape (..., 3, 3)")
    a = M
    C = torch.empty_like(a)
    C[..., 0, 0] = a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]
    C[..., 0, 1] = a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]
    C[..., 0, 2] = a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]

    C[..., 1, 0] = a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2]
    C[..., 1, 1] = a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0]
    C[..., 1, 2] = a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1]

    C[..., 2, 0] = a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1]
    C[..., 2, 1] = a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2]
    C[..., 2, 2] = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    return C


def _signed_inv_cuberoot(det: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    return sign * torch.clamp(det.abs(), min=eps).pow(-1.0 / 3.0)


def deep_coro_forward_torch(
    M: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable PyTorch implementation of the Deep CORO forward pass."""
    if M.shape[-2:] != (3, 3):
        raise ValueError("M must have shape (..., 3, 3)")
    if alpha.ndim != 1 or beta.ndim != 1 or alpha.numel() != beta.numel():
        raise ValueError("alpha and beta must be one-dimensional tensors with equal length")
    X = M
    for aa, bb in zip(alpha, beta):
        Y = aa * X + bb * cofactor3(X)
        det = torch.linalg.det(Y)
        rho = _signed_inv_cuberoot(det, eps)
        X = rho[..., None, None] * Y
    return X


def deep_coro_forward(
    M: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-12,
    use_extension: bool = True,
) -> torch.Tensor:
    """Run Deep CORO with the extension when available, otherwise fallback.

    The custom extension is forward-only.  In training code, call
    `deep_coro_forward_torch` or use `DeepCOROModule(..., use_extension=False)`.
    """
    if use_extension and deep_coro_ext is not None and not torch.is_grad_enabled():
        return deep_coro_ext.deep_coro_forward(M.contiguous(), alpha.contiguous(), beta.contiguous(), eps)
    if use_extension and deep_coro_ext is not None and not M.requires_grad:
        return deep_coro_ext.deep_coro_forward(M.contiguous(), alpha.contiguous(), beta.contiguous(), eps)
    return deep_coro_forward_torch(M, alpha, beta, eps)


def svd_project_so3(M: torch.Tensor) -> torch.Tensor:
    """Frobenius nearest rotation projection via batched SVD."""
    U, _, Vh = torch.linalg.svd(M)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(M.shape[:-2] + (3, 3)).clone()
    det = torch.linalg.det(U @ Vh)
    D[..., 2, 2] = det
    return U @ D @ Vh


def random_rotations(batch: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample random rotations by QR decomposition."""
    if device is None:
        device = torch.device("cpu")
    A = torch.randn(batch, 3, 3, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    neg = torch.linalg.det(Q) < 0
    Q[neg, :, 0] *= -1
    return Q


def rotation_error_deg(R: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error in degrees."""
    rel_trace = torch.einsum("...ij,...ij->...", R, R_ref)
    c = torch.clamp((rel_trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(c))
