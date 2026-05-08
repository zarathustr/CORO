"""Activation functions used by Zhang neural networks."""

from __future__ import annotations

import numpy as np


def linear_activation(error: np.ndarray) -> np.ndarray:
    """Linear activation f(e)=e."""
    return np.asarray(error, dtype=float)


def sigmoid_activation(error: np.ndarray, xi: float = 4.0) -> np.ndarray:
    """Bipolar sigmoid activation from Zhang 2007."""
    error = np.asarray(error, dtype=float)
    return np.tanh(0.5 * xi * error)


def power_activation(error: np.ndarray, power: int = 3) -> np.ndarray:
    """Odd power activation."""
    error = np.asarray(error, dtype=float)
    return np.power(error, power)


def power_sigmoid_activation(error: np.ndarray, power: int = 3, xi: float = 4.0) -> np.ndarray:
    """Power-sigmoid activation from Zhang 2007.

    For |e| <= 1 it uses e^p; otherwise it uses the scaled bipolar sigmoid.
    """
    error = np.asarray(error, dtype=float)
    output = np.empty_like(error)
    mask = np.abs(error) <= 1.0
    output[mask] = np.power(error[mask], power)
    scale = (1.0 + np.exp(-xi)) / (1.0 - np.exp(-xi))
    output[~mask] = scale * np.tanh(0.5 * xi * error[~mask])
    return output


def matrix_activation(
    error: np.ndarray,
    kind: str = "linear",
    power: int = 3,
    xi: float = 4.0,
) -> np.ndarray:
    """Dispatch helper for matrix-wise activation."""
    if kind == "linear":
        return linear_activation(error)
    if kind == "sigmoid":
        return sigmoid_activation(error, xi=xi)
    if kind == "power":
        return power_activation(error, power=power)
    if kind == "power_sigmoid":
        return power_sigmoid_activation(error, power=power, xi=xi)
    raise ValueError("Unknown activation kind: {0}".format(kind))
