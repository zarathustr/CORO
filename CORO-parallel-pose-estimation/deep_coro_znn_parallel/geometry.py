"""SO(3) / SE(3) geometry helpers.

The implementation follows MATLAB-style column-major vectorization whenever a
Kronecker linearization is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def vec_f(matrix: np.ndarray) -> np.ndarray:
    """Column-major vectorization."""
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def mat_f(vector: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Inverse of vec_f."""
    return np.asarray(vector, dtype=float).reshape((rows, cols), order="F")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random 3 x 3 rotation."""
    return Rotation.random(random_state=rng).as_matrix()


def random_se3(rng: np.random.Generator, translation_scale: float = 0.5) -> np.ndarray:
    """Random SE(3) transform with unconstrained attitude."""
    transform = np.eye(4)
    transform[:3, :3] = random_rotation(rng)
    transform[:3, 3] = translation_scale * rng.standard_normal(3)
    return transform


def random_small_se3(
    rng: np.random.Generator,
    rotation_deg: float = 20.0,
    translation_scale: float = 0.2,
) -> np.ndarray:
    """Near-identity SE(3) transform.

    This better matches calibration practice, where the sought extrinsics are
    often not arbitrarily far from a nominal frame.
    """
    axis = rng.standard_normal(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        rotation = np.eye(3)
    else:
        axis /= norm
        angle = np.deg2rad(rotation_deg) * float(rng.random())
        rotation = Rotation.from_rotvec(axis * angle).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_scale * rng.standard_normal(3)
    return transform


def se3_inverse(transform: np.ndarray) -> np.ndarray:
    """Inverse of SE(3)."""
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def add_rotation_noise(rotation: np.ndarray, rng: np.random.Generator, noise_deg: float) -> np.ndarray:
    """Right-multiply by a small random rotation."""
    axis = rng.standard_normal(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        delta = np.eye(3)
    else:
        axis /= norm
        angle = np.deg2rad(noise_deg) * float(abs(rng.standard_normal()))
        delta = Rotation.from_rotvec(axis * angle).as_matrix()
    return np.asarray(rotation, dtype=float) @ delta


def add_se3_noise(
    transform: np.ndarray,
    rng: np.random.Generator,
    noise_deg: float = 0.5,
    translation_sigma: float = 0.005,
) -> np.ndarray:
    """Right-multiply by a small SE(3) perturbation."""
    delta = np.eye(4)
    delta[:3, :3] = add_rotation_noise(np.eye(3), rng, noise_deg=noise_deg)
    delta[:3, 3] = translation_sigma * rng.standard_normal(3)
    return np.asarray(transform, dtype=float) @ delta


def split_se3(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an SE(3) matrix into rotation and translation."""
    transform = np.asarray(transform, dtype=float)
    return transform[:3, :3], transform[:3, 3].copy()


def compose_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Compose SE(3) from rotation and translation."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def rotation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Geodesic SO(3) error in degrees."""
    relative = np.asarray(reference, dtype=float).T @ np.asarray(estimate, dtype=float)
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def translation_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Euclidean translation error."""
    return float(np.linalg.norm(np.asarray(reference, dtype=float)[:3, 3] - np.asarray(estimate, dtype=float)[:3, 3]))


@dataclass
class PoseError:
    rotation_deg: float
    translation: float


def pose_error(reference: np.ndarray, estimate: np.ndarray) -> PoseError:
    """Combined SE(3) pose error."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return PoseError(
        rotation_deg=rotation_error_deg(reference[:3, :3], estimate[:3, :3]),
        translation=translation_error(reference, estimate),
    )
