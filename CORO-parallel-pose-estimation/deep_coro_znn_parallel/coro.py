"""Completely Rational Orthonormalizer (CORO) for SO(3)."""

from __future__ import annotations

import numpy as np


def adjugate3(matrix: np.ndarray) -> np.ndarray:
    """Adjugate of a 3 x 3 matrix via column-wise cross products."""
    matrix = np.asarray(matrix, dtype=float)
    c1 = matrix[:, 0]
    c2 = matrix[:, 1]
    c3 = matrix[:, 2]
    return np.column_stack((np.cross(c2, c3), np.cross(c3, c1), np.cross(c1, c2)))


def project_so_svd(matrix: np.ndarray) -> np.ndarray:
    """Nearest rotation via SVD."""
    u_matrix, _, vt_matrix = np.linalg.svd(np.asarray(matrix, dtype=float))
    sign_fix = np.eye(3)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        sign_fix[-1, -1] = -1.0
    return u_matrix @ sign_fix @ vt_matrix


def coro_step(
    matrix: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """One 3-D CORO step."""
    matrix = np.asarray(matrix, dtype=float)
    fro_sq = float(np.sum(matrix * matrix))
    rho = 2.0 / (1.0 + fro_sq + eps)
    return rho * (alpha * matrix + beta * adjugate3(matrix))


def project_so_coro(
    matrix: np.ndarray,
    num_iters: int = 6,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project to SO(3) with repeated CORO updates."""
    current = np.asarray(matrix, dtype=float).copy()
    scale = float(np.max(np.abs(current)))
    if scale < eps:
        current = np.eye(3)
    else:
        current /= scale
    if np.linalg.det(current) < 0.0:
        current[:, -1] *= -1.0
    for _ in range(int(num_iters)):
        current = coro_step(current, alpha=alpha, beta=beta, eps=eps)
    if np.linalg.det(current) < 0.0:
        current[:, -1] *= -1.0
    return current


def project_so(matrix: np.ndarray, method: str = "coro", num_iters: int = 6) -> np.ndarray:
    """Dispatch helper for SO(3) projection."""
    if method == "svd":
        return project_so_svd(matrix)
    if method == "coro":
        return project_so_coro(matrix, num_iters=num_iters)
    raise ValueError("Unknown SO(3) projector: {0}".format(method))


def project_se3_rotation(transform: np.ndarray, method: str = "coro") -> np.ndarray:
    """Project the 3 x 3 rotation block of an SE(3) matrix."""
    result = np.asarray(transform, dtype=float).copy()
    result[:3, :3] = project_so(result[:3, :3], method=method)
    result[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return result
