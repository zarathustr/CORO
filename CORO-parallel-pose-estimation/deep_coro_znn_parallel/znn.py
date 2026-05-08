"""Generic Zhang-neural-network helpers.

The Zhang 2007 paper formulates the Sylvester solver from the matrix-valued
error E(t)=A(t)X(t)-X(t)B(t)+C(t) and imposes the implicit dynamics

    dE/dt = -Gamma F(E),

which leads to

    A(t) Xdot - Xdot B(t) = -Adot(t) X + X Bdot(t) - Cdot(t) - gamma F(E).

For the calibration problems here, the measurements are static, so the explicit
A-dot/B-dot/C-dot terms vanish. The resulting implicit linear system is solved
at each Euler step by a least-squares solve in vectorized form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .activations import matrix_activation
from .geometry import vec_f


@dataclass
class ZNNStepResult:
    derivative: np.ndarray
    residual_norm: float
    mass_condition: float


@dataclass
class GNNStepResult:
    derivative: np.ndarray
    residual_norm: float


def solve_implicit_znn_step(
    mass_matrix: np.ndarray,
    residual_blocks: Sequence[np.ndarray],
    gamma: float,
    activation: str = "linear",
    power: int = 3,
    xi: float = 4.0,
) -> ZNNStepResult:
    """Solve M(theta) theta_dot = -gamma F(E) for one ZNN Euler step."""
    rhs_parts = []
    residual_norm_sq = 0.0
    for block in residual_blocks:
        activated = matrix_activation(block, kind=activation, power=power, xi=xi)
        rhs_parts.append(-gamma * vec_f(activated))
        residual_norm_sq += float(np.sum(block * block))
    rhs = np.concatenate(rhs_parts, axis=0)
    derivative, _, _, _ = np.linalg.lstsq(np.asarray(mass_matrix, dtype=float), rhs, rcond=None)
    try:
        condition = float(np.linalg.cond(np.asarray(mass_matrix, dtype=float)))
    except np.linalg.LinAlgError:
        condition = float("inf")
    return ZNNStepResult(
        derivative=derivative,
        residual_norm=float(np.sqrt(residual_norm_sq)),
        mass_condition=condition,
    )


def build_gradient_descent_step(
    gradient: np.ndarray,
    residual_blocks: Sequence[np.ndarray],
) -> GNNStepResult:
    """Package an explicit gradient-based neural-network step."""
    residual_norm_sq = 0.0
    for block in residual_blocks:
        residual_norm_sq += float(np.sum(block * block))
    return GNNStepResult(
        derivative=np.asarray(gradient, dtype=float),
        residual_norm=float(np.sqrt(residual_norm_sq)),
    )
