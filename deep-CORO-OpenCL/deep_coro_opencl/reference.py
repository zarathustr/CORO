from __future__ import annotations

import numpy as np


def cofactor3_np(M: np.ndarray) -> np.ndarray:
    """Batched 3x3 cofactor map for arrays with shape (..., 3, 3)."""
    A = np.asarray(M)
    if A.shape[-2:] != (3, 3):
        raise ValueError("cofactor3_np expects shape (...,3,3)")
    C = np.empty_like(A)
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


def det3_np(M: np.ndarray) -> np.ndarray:
    A = np.asarray(M)
    return (
        A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
        - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
        + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
    )


def deep_coro_forward_np(
    M: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Vectorized NumPy Deep CORO forward pass for SO(3).

    Parameters
    ----------
    M:
        Batch of raw matrices with shape (B,3,3).
    alpha, beta:
        One-dimensional arrays with one coefficient per CORO layer.
    eps:
        Determinant clamp used in the radial scale.
    """
    X = np.asarray(M).copy()
    alpha = np.asarray(alpha, dtype=X.dtype)
    beta = np.asarray(beta, dtype=X.dtype)
    if X.ndim != 3 or X.shape[1:] != (3, 3):
        raise ValueError("M must have shape (B,3,3)")
    if alpha.ndim != 1 or beta.ndim != 1 or alpha.size != beta.size:
        raise ValueError("alpha and beta must be 1-D arrays of the same length")
    for aa, bb in zip(alpha, beta):
        Y = aa * X + bb * cofactor3_np(X)
        det = det3_np(Y)
        sign = np.where(det >= 0, 1.0, -1.0).astype(X.dtype)
        rho = sign * np.maximum(np.abs(det), eps).astype(X.dtype) ** np.asarray(-1.0 / 3.0, dtype=X.dtype)
        X = rho[:, None, None] * Y
    return X


def svd_project_so3(M: np.ndarray) -> np.ndarray:
    """Frobenius nearest rotation projection through batched SVD."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    det = np.linalg.det(R)
    bad = det < 0
    if np.any(bad):
        U2 = U.copy()
        U2[bad, :, 2] *= -1.0
        R = U2 @ Vt
    return R.astype(M.dtype, copy=False)


def random_rotations(batch: int, rng: np.random.Generator, dtype=np.float32) -> np.ndarray:
    A = rng.standard_normal((batch, 3, 3)).astype(dtype)
    Q = np.empty_like(A)
    for i in range(batch):
        q, r = np.linalg.qr(A[i].astype(np.float64))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        Q[i] = q.astype(dtype)
    return Q


def make_distorted_rotations(batch: int, rng: np.random.Generator, dtype=np.float32):
    R = random_rotations(batch, rng, dtype=dtype)
    log_scales = (0.60 * rng.standard_normal((batch, 3))).astype(dtype)
    scales = np.exp(log_scales).astype(dtype)
    noise = (0.025 * rng.standard_normal((batch, 3, 3))).astype(dtype)
    M = np.matmul(R, np.eye(3, dtype=dtype)[None, :, :] * scales[:, None, :]) + noise
    return M.astype(dtype), R.astype(dtype)


def rotation_error_deg(R: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    tr = np.einsum("bij,bij->b", R, R_ref)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def orthogonality_error(R: np.ndarray) -> np.ndarray:
    I = np.eye(3, dtype=R.dtype)
    G = np.matmul(np.swapaxes(R, -1, -2), R) - I[None, :, :]
    return np.linalg.norm(G, axis=(-2, -1))
