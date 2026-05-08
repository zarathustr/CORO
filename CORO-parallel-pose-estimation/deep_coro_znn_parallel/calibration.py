"""Corrected parallel calibration solvers based on Zhang neural dynamics.

This module implements three SE(3) calibration variants as *Sylvester-like*
implicit dynamical systems:

1. Hand-eye:                A_i X = X B_i
2. Robot-world/hand-eye:    A_i X = Y B_i
3. Multi-frame calibration: A_i X B_i = Y C_i Z

For each problem, the residual matrix is differentiated, the derivative is set
according to the Zhang-neural-network rule dE/dt = -gamma F(E), and the
resulting linear system in the state derivative is solved by least squares.
After each Euler step, the rotation blocks are projected back to SO(3) with
either SVD or CORO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .activations import matrix_activation
from .coro import project_so
from .geometry import (
    PoseError,
    add_se3_noise,
    compose_se3,
    pose_error,
    random_small_se3,
    se3_inverse,
    split_se3,
    vec_f,
    mat_f,
)
from .znn import build_gradient_descent_step, solve_implicit_znn_step


Projector = str
Dynamics = str


@dataclass
class HandEyeInstance:
    motions_a: List[np.ndarray]
    motions_b: List[np.ndarray]
    x_true: np.ndarray


@dataclass
class RWHEInstance:
    motions_a: List[np.ndarray]
    motions_b: List[np.ndarray]
    x_true: np.ndarray
    y_true: np.ndarray


@dataclass
class AXBYCZInstance:
    motions_a: List[np.ndarray]
    motions_b: List[np.ndarray]
    motions_c: List[np.ndarray]
    x_true: np.ndarray
    y_true: np.ndarray
    z_true: np.ndarray


@dataclass
class SolverResultSingle:
    estimate: np.ndarray
    history: List[Dict[str, float]]


@dataclass
class SolverResultDouble:
    x_estimate: np.ndarray
    y_estimate: np.ndarray
    history: List[Dict[str, float]]


@dataclass
class SolverResultTriple:
    x_estimate: np.ndarray
    y_estimate: np.ndarray
    z_estimate: np.ndarray
    history: List[Dict[str, float]]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_hand_eye_instance(
    rng: np.random.Generator,
    num_pairs: int = 6,
    measurement_rot_deg: float = 0.5,
    measurement_translation_sigma: float = 0.005,
) -> HandEyeInstance:
    """Synthetic A_i X = X B_i with near-identity extrinsics."""
    x_true = random_small_se3(rng)
    motions_a: List[np.ndarray] = []
    motions_b: List[np.ndarray] = []
    for _ in range(num_pairs):
        motion_a = random_small_se3(rng, rotation_deg=60.0, translation_scale=0.4)
        motion_b = se3_inverse(x_true) @ motion_a @ x_true
        motions_a.append(add_se3_noise(motion_a, rng, measurement_rot_deg, measurement_translation_sigma))
        motions_b.append(add_se3_noise(motion_b, rng, measurement_rot_deg, measurement_translation_sigma))
    return HandEyeInstance(motions_a=motions_a, motions_b=motions_b, x_true=x_true)



def generate_rwhe_instance(
    rng: np.random.Generator,
    num_pairs: int = 6,
    measurement_rot_deg: float = 0.5,
    measurement_translation_sigma: float = 0.005,
) -> RWHEInstance:
    """Synthetic A_i X = Y B_i with near-identity extrinsics."""
    x_true = random_small_se3(rng)
    y_true = random_small_se3(rng)
    motions_a: List[np.ndarray] = []
    motions_b: List[np.ndarray] = []
    for _ in range(num_pairs):
        motion_a = random_small_se3(rng, rotation_deg=60.0, translation_scale=0.4)
        motion_b = se3_inverse(y_true) @ motion_a @ x_true
        motions_a.append(add_se3_noise(motion_a, rng, measurement_rot_deg, measurement_translation_sigma))
        motions_b.append(add_se3_noise(motion_b, rng, measurement_rot_deg, measurement_translation_sigma))
    return RWHEInstance(motions_a=motions_a, motions_b=motions_b, x_true=x_true, y_true=y_true)



def generate_axbycz_instance(
    rng: np.random.Generator,
    num_pairs: int = 8,
    measurement_rot_deg: float = 0.5,
    measurement_translation_sigma: float = 0.005,
) -> AXBYCZInstance:
    """Synthetic A_i X B_i = Y C_i Z with near-identity extrinsics."""
    x_true = random_small_se3(rng)
    y_true = random_small_se3(rng)
    z_true = random_small_se3(rng)
    motions_a: List[np.ndarray] = []
    motions_b: List[np.ndarray] = []
    motions_c: List[np.ndarray] = []
    for _ in range(num_pairs):
        motion_a = random_small_se3(rng, rotation_deg=60.0, translation_scale=0.4)
        motion_b = random_small_se3(rng, rotation_deg=60.0, translation_scale=0.4)
        motion_c = se3_inverse(y_true) @ motion_a @ x_true @ motion_b @ se3_inverse(z_true)
        motions_a.append(add_se3_noise(motion_a, rng, measurement_rot_deg, measurement_translation_sigma))
        motions_b.append(add_se3_noise(motion_b, rng, measurement_rot_deg, measurement_translation_sigma))
        motions_c.append(add_se3_noise(motion_c, rng, measurement_rot_deg, measurement_translation_sigma))
    return AXBYCZInstance(
        motions_a=motions_a,
        motions_b=motions_b,
        motions_c=motions_c,
        x_true=x_true,
        y_true=y_true,
        z_true=z_true,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def hand_eye_loss(motions_a: Sequence[np.ndarray], motions_b: Sequence[np.ndarray], x_estimate: np.ndarray) -> float:
    residuals = [motion_a @ x_estimate - x_estimate @ motion_b for motion_a, motion_b in zip(motions_a, motions_b)]
    return float(np.mean([np.sum(residual * residual) for residual in residuals]))



def rwhe_loss(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    x_estimate: np.ndarray,
    y_estimate: np.ndarray,
) -> float:
    residuals = [motion_a @ x_estimate - y_estimate @ motion_b for motion_a, motion_b in zip(motions_a, motions_b)]
    return float(np.mean([np.sum(residual * residual) for residual in residuals]))



def axbycz_loss(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    motions_c: Sequence[np.ndarray],
    x_estimate: np.ndarray,
    y_estimate: np.ndarray,
    z_estimate: np.ndarray,
) -> float:
    residuals = [
        motion_a @ x_estimate @ motion_b - y_estimate @ motion_c @ z_estimate
        for motion_a, motion_b, motion_c in zip(motions_a, motions_b, motions_c)
    ]
    return float(np.mean([np.sum(residual * residual) for residual in residuals]))


# ---------------------------------------------------------------------------
# Problem-specific residuals and mass matrices
# ---------------------------------------------------------------------------


def _project_rotation(rotation: np.ndarray, projector: Projector) -> np.ndarray:
    return project_so(rotation, method=projector)


# ---- Hand-eye ----------------------------------------------------------------


def _hand_eye_residual_split(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    residual_rot: List[np.ndarray] = []
    residual_trans: List[np.ndarray] = []
    for motion_a, motion_b in zip(motions_a, motions_b):
        rotation_a, translation_a = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        residual_rot.append(rotation_a @ rotation_x - rotation_x @ rotation_b)
        residual_trans.append(rotation_a @ translation_x + translation_a - rotation_x @ translation_b - translation_x)
    return residual_rot, residual_trans



def _hand_eye_znn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    identity3 = np.eye(3)
    rows: List[np.ndarray] = []
    residual_blocks: List[np.ndarray] = []
    residual_rot, residual_trans = _hand_eye_residual_split(motions_a, motions_b, rotation_x, translation_x)
    for motion_a, motion_b, residual_r, residual_t in zip(motions_a, motions_b, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        mass_rot = np.hstack((np.kron(identity3, rotation_a) - np.kron(rotation_b.T, identity3), np.zeros((9, 3))))
        mass_trans = np.hstack((-np.kron(translation_b.reshape(1, 3), identity3), rotation_a - identity3))
        rows.append(np.vstack((mass_rot, mass_trans)))
        residual_blocks.append(residual_r)
        residual_blocks.append(residual_t.reshape(3, 1))
    step = solve_implicit_znn_step(np.vstack(rows), residual_blocks, gamma=gamma, activation="linear")
    derivative_rotation = mat_f(step.derivative[:9], 3, 3)
    derivative_translation = step.derivative[9:12]
    return derivative_rotation, derivative_translation, step.residual_norm, step.mass_condition



def _hand_eye_gnn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    derivative_rotation = np.zeros((3, 3))
    derivative_translation = np.zeros(3)
    residual_rot, residual_trans = _hand_eye_residual_split(motions_a, motions_b, rotation_x, translation_x)
    count = float(len(motions_a))
    for motion_a, motion_b, residual_r, residual_t in zip(motions_a, motions_b, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        derivative_rotation += -gamma * (rotation_a.T @ residual_r - residual_r @ rotation_b.T - np.outer(residual_t, translation_b)) / count
        derivative_translation += -gamma * ((rotation_a - np.eye(3)).T @ residual_t) / count
    step = build_gradient_descent_step(np.concatenate((vec_f(derivative_rotation), derivative_translation)), [residual_r for residual_r in residual_rot] + [residual_t.reshape(3, 1) for residual_t in residual_trans])
    return derivative_rotation, derivative_translation, step.residual_norm


# ---- Robot-world/hand-eye ----------------------------------------------------


def _rwhe_residual_split(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    residual_rot: List[np.ndarray] = []
    residual_trans: List[np.ndarray] = []
    for motion_a, motion_b in zip(motions_a, motions_b):
        rotation_a, translation_a = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        residual_rot.append(rotation_a @ rotation_x - rotation_y @ rotation_b)
        residual_trans.append(rotation_a @ translation_x + translation_a - rotation_y @ translation_b - translation_y)
    return residual_rot, residual_trans



def _rwhe_znn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    identity3 = np.eye(3)
    rows: List[np.ndarray] = []
    residual_blocks: List[np.ndarray] = []
    residual_rot, residual_trans = _rwhe_residual_split(motions_a, motions_b, rotation_x, translation_x, rotation_y, translation_y)
    for motion_a, motion_b, residual_r, residual_t in zip(motions_a, motions_b, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        mass_rot = np.hstack((
            np.kron(identity3, rotation_a),
            np.zeros((9, 3)),
            -np.kron(rotation_b.T, identity3),
            np.zeros((9, 3)),
        ))
        mass_trans = np.hstack((
            np.zeros((3, 9)),
            rotation_a,
            -np.kron(translation_b.reshape(1, 3), identity3),
            -identity3,
        ))
        rows.append(np.vstack((mass_rot, mass_trans)))
        residual_blocks.append(residual_r)
        residual_blocks.append(residual_t.reshape(3, 1))
    step = solve_implicit_znn_step(np.vstack(rows), residual_blocks, gamma=gamma, activation="linear")
    derivative_rotation_x = mat_f(step.derivative[:9], 3, 3)
    derivative_translation_x = step.derivative[9:12]
    derivative_rotation_y = mat_f(step.derivative[12:21], 3, 3)
    derivative_translation_y = step.derivative[21:24]
    return (
        derivative_rotation_x,
        derivative_translation_x,
        derivative_rotation_y,
        derivative_translation_y,
        step.residual_norm,
        step.mass_condition,
    )



def _rwhe_gnn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    derivative_rotation_x = np.zeros((3, 3))
    derivative_translation_x = np.zeros(3)
    derivative_rotation_y = np.zeros((3, 3))
    derivative_translation_y = np.zeros(3)
    residual_rot, residual_trans = _rwhe_residual_split(motions_a, motions_b, rotation_x, translation_x, rotation_y, translation_y)
    count = float(len(motions_a))
    for motion_a, motion_b, residual_r, residual_t in zip(motions_a, motions_b, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        derivative_rotation_x += -gamma * (rotation_a.T @ residual_r) / count
        derivative_translation_x += -gamma * (rotation_a.T @ residual_t) / count
        derivative_rotation_y += +gamma * (residual_r @ rotation_b.T + np.outer(residual_t, translation_b)) / count
        derivative_translation_y += +gamma * residual_t / count
    step = build_gradient_descent_step(
        np.concatenate((vec_f(derivative_rotation_x), derivative_translation_x, vec_f(derivative_rotation_y), derivative_translation_y)),
        [residual_r for residual_r in residual_rot] + [residual_t.reshape(3, 1) for residual_t in residual_trans],
    )
    return derivative_rotation_x, derivative_translation_x, derivative_rotation_y, derivative_translation_y, step.residual_norm


# ---- A X B = Y C Z -----------------------------------------------------------


def _axbycz_residual_split(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    motions_c: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
    rotation_z: np.ndarray,
    translation_z: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    residual_rot: List[np.ndarray] = []
    residual_trans: List[np.ndarray] = []
    for motion_a, motion_b, motion_c in zip(motions_a, motions_b, motions_c):
        rotation_a, translation_a = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        rotation_c, translation_c = split_se3(motion_c)
        residual_rot.append(rotation_a @ rotation_x @ rotation_b - rotation_y @ rotation_c @ rotation_z)
        residual_trans.append(
            rotation_a @ rotation_x @ translation_b
            + rotation_a @ translation_x
            + translation_a
            - (rotation_y @ rotation_c @ translation_z + rotation_y @ translation_c + translation_y)
        )
    return residual_rot, residual_trans



def _axbycz_znn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    motions_c: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
    rotation_z: np.ndarray,
    translation_z: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    identity3 = np.eye(3)
    rows: List[np.ndarray] = []
    residual_blocks: List[np.ndarray] = []
    residual_rot, residual_trans = _axbycz_residual_split(
        motions_a,
        motions_b,
        motions_c,
        rotation_x,
        translation_x,
        rotation_y,
        translation_y,
        rotation_z,
        translation_z,
    )
    for motion_a, motion_b, motion_c, residual_r, residual_t in zip(motions_a, motions_b, motions_c, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        rotation_c, translation_c = split_se3(motion_c)
        mass_rot = np.hstack((
            np.kron(rotation_b.T, rotation_a),
            np.zeros((9, 3)),
            -np.kron((rotation_c @ rotation_z).T, identity3),
            np.zeros((9, 3)),
            -np.kron(identity3, rotation_y @ rotation_c),
            np.zeros((9, 3)),
        ))
        mass_trans = np.hstack((
            np.kron(translation_b.reshape(1, 3), rotation_a),
            rotation_a,
            -np.kron((rotation_c @ translation_z + translation_c).reshape(1, 3), identity3),
            -identity3,
            np.zeros((3, 9)),
            -(rotation_y @ rotation_c),
        ))
        rows.append(np.vstack((mass_rot, mass_trans)))
        residual_blocks.append(residual_r)
        residual_blocks.append(residual_t.reshape(3, 1))
    step = solve_implicit_znn_step(np.vstack(rows), residual_blocks, gamma=gamma, activation="linear")
    derivative_rotation_x = mat_f(step.derivative[:9], 3, 3)
    derivative_translation_x = step.derivative[9:12]
    derivative_rotation_y = mat_f(step.derivative[12:21], 3, 3)
    derivative_translation_y = step.derivative[21:24]
    derivative_rotation_z = mat_f(step.derivative[24:33], 3, 3)
    derivative_translation_z = step.derivative[33:36]
    return (
        derivative_rotation_x,
        derivative_translation_x,
        derivative_rotation_y,
        derivative_translation_y,
        derivative_rotation_z,
        derivative_translation_z,
        step.residual_norm,
        step.mass_condition,
    )



def _axbycz_gnn_step(
    motions_a: Sequence[np.ndarray],
    motions_b: Sequence[np.ndarray],
    motions_c: Sequence[np.ndarray],
    rotation_x: np.ndarray,
    translation_x: np.ndarray,
    rotation_y: np.ndarray,
    translation_y: np.ndarray,
    rotation_z: np.ndarray,
    translation_z: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    derivative_rotation_x = np.zeros((3, 3))
    derivative_translation_x = np.zeros(3)
    derivative_rotation_y = np.zeros((3, 3))
    derivative_translation_y = np.zeros(3)
    derivative_rotation_z = np.zeros((3, 3))
    derivative_translation_z = np.zeros(3)
    residual_rot, residual_trans = _axbycz_residual_split(
        motions_a,
        motions_b,
        motions_c,
        rotation_x,
        translation_x,
        rotation_y,
        translation_y,
        rotation_z,
        translation_z,
    )
    count = float(len(motions_a))
    for motion_a, motion_b, motion_c, residual_r, residual_t in zip(motions_a, motions_b, motions_c, residual_rot, residual_trans):
        rotation_a, _ = split_se3(motion_a)
        rotation_b, translation_b = split_se3(motion_b)
        rotation_c, translation_c = split_se3(motion_c)
        derivative_rotation_x += -gamma * (rotation_a.T @ residual_r @ rotation_b.T + rotation_a.T @ np.outer(residual_t, translation_b)) / count
        derivative_translation_x += -gamma * (rotation_a.T @ residual_t) / count
        derivative_rotation_y += +gamma * (residual_r @ (rotation_c @ rotation_z).T + np.outer(residual_t, rotation_c @ translation_z + translation_c)) / count
        derivative_translation_y += +gamma * residual_t / count
        derivative_rotation_z += +gamma * ((rotation_y @ rotation_c).T @ residual_r) / count
        derivative_translation_z += +gamma * ((rotation_y @ rotation_c).T @ residual_t) / count
    step = build_gradient_descent_step(
        np.concatenate((
            vec_f(derivative_rotation_x),
            derivative_translation_x,
            vec_f(derivative_rotation_y),
            derivative_translation_y,
            vec_f(derivative_rotation_z),
            derivative_translation_z,
        )),
        [residual_r for residual_r in residual_rot] + [residual_t.reshape(3, 1) for residual_t in residual_trans],
    )
    return (
        derivative_rotation_x,
        derivative_translation_x,
        derivative_rotation_y,
        derivative_translation_y,
        derivative_rotation_z,
        derivative_translation_z,
        step.residual_norm,
    )


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------


def solve_hand_eye(
    instance: HandEyeInstance,
    dynamics: Dynamics = "znn",
    projector: Projector = "coro",
    dt: float = 0.05,
    steps: int = 100,
    gamma: Optional[float] = None,
) -> SolverResultSingle:
    """Solve A_i X = X B_i."""
    gamma_value = 1.0 if gamma is None and dynamics == "znn" else (0.05 if gamma is None else gamma)
    rotation_x = np.eye(3)
    translation_x = np.zeros(3)
    history: List[Dict[str, float]] = []
    for step_index in range(int(steps)):
        if dynamics == "znn":
            derivative_rotation, derivative_translation, residual_norm, mass_condition = _hand_eye_znn_step(
                instance.motions_a,
                instance.motions_b,
                rotation_x,
                translation_x,
                gamma=gamma_value,
            )
        elif dynamics == "gnn":
            derivative_rotation, derivative_translation, residual_norm = _hand_eye_gnn_step(
                instance.motions_a,
                instance.motions_b,
                rotation_x,
                translation_x,
                gamma=gamma_value,
            )
            mass_condition = float("nan")
        else:
            raise ValueError("Unknown dynamics: {0}".format(dynamics))
        rotation_x = _project_rotation(rotation_x + dt * derivative_rotation, projector)
        translation_x = translation_x + dt * derivative_translation
        estimate = compose_se3(rotation_x, translation_x)
        history.append(
            {
                "iteration": float(step_index),
                "loss": hand_eye_loss(instance.motions_a, instance.motions_b, estimate),
                "residual_norm": residual_norm,
                "mass_condition": mass_condition,
            }
        )
    return SolverResultSingle(estimate=compose_se3(rotation_x, translation_x), history=history)



def solve_rwhe(
    instance: RWHEInstance,
    dynamics: Dynamics = "znn",
    projector: Projector = "coro",
    dt: float = 0.05,
    steps: int = 100,
    gamma: Optional[float] = None,
) -> SolverResultDouble:
    """Solve A_i X = Y B_i."""
    gamma_value = 1.0 if gamma is None and dynamics == "znn" else (0.05 if gamma is None else gamma)
    rotation_x = np.eye(3)
    translation_x = np.zeros(3)
    rotation_y = np.eye(3)
    translation_y = np.zeros(3)
    history: List[Dict[str, float]] = []
    for step_index in range(int(steps)):
        if dynamics == "znn":
            (
                derivative_rotation_x,
                derivative_translation_x,
                derivative_rotation_y,
                derivative_translation_y,
                residual_norm,
                mass_condition,
            ) = _rwhe_znn_step(
                instance.motions_a,
                instance.motions_b,
                rotation_x,
                translation_x,
                rotation_y,
                translation_y,
                gamma=gamma_value,
            )
        elif dynamics == "gnn":
            (
                derivative_rotation_x,
                derivative_translation_x,
                derivative_rotation_y,
                derivative_translation_y,
                residual_norm,
            ) = _rwhe_gnn_step(
                instance.motions_a,
                instance.motions_b,
                rotation_x,
                translation_x,
                rotation_y,
                translation_y,
                gamma=gamma_value,
            )
            mass_condition = float("nan")
        else:
            raise ValueError("Unknown dynamics: {0}".format(dynamics))
        rotation_x = _project_rotation(rotation_x + dt * derivative_rotation_x, projector)
        translation_x = translation_x + dt * derivative_translation_x
        rotation_y = _project_rotation(rotation_y + dt * derivative_rotation_y, projector)
        translation_y = translation_y + dt * derivative_translation_y
        x_estimate = compose_se3(rotation_x, translation_x)
        y_estimate = compose_se3(rotation_y, translation_y)
        history.append(
            {
                "iteration": float(step_index),
                "loss": rwhe_loss(instance.motions_a, instance.motions_b, x_estimate, y_estimate),
                "residual_norm": residual_norm,
                "mass_condition": mass_condition,
            }
        )
    return SolverResultDouble(x_estimate=compose_se3(rotation_x, translation_x), y_estimate=compose_se3(rotation_y, translation_y), history=history)



def solve_axbycz(
    instance: AXBYCZInstance,
    dynamics: Dynamics = "znn",
    projector: Projector = "coro",
    dt: float = 0.05,
    steps: int = 150,
    gamma: Optional[float] = None,
) -> SolverResultTriple:
    """Solve A_i X B_i = Y C_i Z."""
    gamma_value = 1.0 if gamma is None and dynamics == "znn" else (0.02 if gamma is None else gamma)
    rotation_x = np.eye(3)
    translation_x = np.zeros(3)
    rotation_y = np.eye(3)
    translation_y = np.zeros(3)
    rotation_z = np.eye(3)
    translation_z = np.zeros(3)
    history: List[Dict[str, float]] = []
    for step_index in range(int(steps)):
        if dynamics == "znn":
            (
                derivative_rotation_x,
                derivative_translation_x,
                derivative_rotation_y,
                derivative_translation_y,
                derivative_rotation_z,
                derivative_translation_z,
                residual_norm,
                mass_condition,
            ) = _axbycz_znn_step(
                instance.motions_a,
                instance.motions_b,
                instance.motions_c,
                rotation_x,
                translation_x,
                rotation_y,
                translation_y,
                rotation_z,
                translation_z,
                gamma=gamma_value,
            )
        elif dynamics == "gnn":
            (
                derivative_rotation_x,
                derivative_translation_x,
                derivative_rotation_y,
                derivative_translation_y,
                derivative_rotation_z,
                derivative_translation_z,
                residual_norm,
            ) = _axbycz_gnn_step(
                instance.motions_a,
                instance.motions_b,
                instance.motions_c,
                rotation_x,
                translation_x,
                rotation_y,
                translation_y,
                rotation_z,
                translation_z,
                gamma=gamma_value,
            )
            mass_condition = float("nan")
        else:
            raise ValueError("Unknown dynamics: {0}".format(dynamics))
        rotation_x = _project_rotation(rotation_x + dt * derivative_rotation_x, projector)
        translation_x = translation_x + dt * derivative_translation_x
        rotation_y = _project_rotation(rotation_y + dt * derivative_rotation_y, projector)
        translation_y = translation_y + dt * derivative_translation_y
        rotation_z = _project_rotation(rotation_z + dt * derivative_rotation_z, projector)
        translation_z = translation_z + dt * derivative_translation_z
        x_estimate = compose_se3(rotation_x, translation_x)
        y_estimate = compose_se3(rotation_y, translation_y)
        z_estimate = compose_se3(rotation_z, translation_z)
        history.append(
            {
                "iteration": float(step_index),
                "loss": axbycz_loss(instance.motions_a, instance.motions_b, instance.motions_c, x_estimate, y_estimate, z_estimate),
                "residual_norm": residual_norm,
                "mass_condition": mass_condition,
            }
        )
    return SolverResultTriple(
        x_estimate=compose_se3(rotation_x, translation_x),
        y_estimate=compose_se3(rotation_y, translation_y),
        z_estimate=compose_se3(rotation_z, translation_z),
        history=history,
    )


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _record_problem_rows(problem: str, method_name: str, rotation_error_value: float, translation_error_value: float, loss_value: float) -> Dict[str, float]:
    return {
        "problem": problem,
        "method": method_name,
        "rotation_error_deg": rotation_error_value,
        "translation_error": translation_error_value,
        "loss": loss_value,
        "success_rot_le_5deg": 1.0 if rotation_error_value <= 5.0 else 0.0,
    }



def benchmark_calibration_suite(
    num_trials: int = 80,
    seed: int = 7,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the corrected benchmark suite.

    Each trial contains one instance of each problem, so num_trials=80 means
    240 problem instances in total.
    """
    rng = np.random.default_rng(seed)
    methods = [
        ("ZNN-SVD", "znn", "svd"),
        ("ZNN-CORO", "znn", "coro"),
        ("GNN-SVD", "gnn", "svd"),
        ("GNN-CORO", "gnn", "coro"),
    ]
    trial_rows: List[Dict[str, float]] = []
    histories: List[Dict[str, float]] = []
    for trial in range(int(num_trials)):
        trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

        hand_eye_instance = generate_hand_eye_instance(trial_rng)
        rwhe_instance = generate_rwhe_instance(trial_rng)
        axbycz_instance = generate_axbycz_instance(trial_rng)

        for method_name, dynamics, projector in methods:
            hand_eye_result = solve_hand_eye(hand_eye_instance, dynamics=dynamics, projector=projector)
            hand_eye_pose_error = pose_error(hand_eye_instance.x_true, hand_eye_result.estimate)
            row = _record_problem_rows(
                "hand_eye",
                method_name,
                hand_eye_pose_error.rotation_deg,
                hand_eye_pose_error.translation,
                hand_eye_loss(hand_eye_instance.motions_a, hand_eye_instance.motions_b, hand_eye_result.estimate),
            )
            row["trial"] = float(trial)
            trial_rows.append(row)
            if trial == 0:
                for item in hand_eye_result.history:
                    histories.append(
                        {
                            "problem": "hand_eye",
                            "method": method_name,
                            "trial": 0.0,
                            "iteration": item["iteration"],
                            "loss": item["loss"],
                            "residual_norm": item["residual_norm"],
                            "mass_condition": item["mass_condition"],
                        }
                    )

            rwhe_result = solve_rwhe(rwhe_instance, dynamics=dynamics, projector=projector)
            rwhe_errors = [pose_error(rwhe_instance.x_true, rwhe_result.x_estimate), pose_error(rwhe_instance.y_true, rwhe_result.y_estimate)]
            row = _record_problem_rows(
                "rwhe",
                method_name,
                max(error.rotation_deg for error in rwhe_errors),
                max(error.translation for error in rwhe_errors),
                rwhe_loss(rwhe_instance.motions_a, rwhe_instance.motions_b, rwhe_result.x_estimate, rwhe_result.y_estimate),
            )
            row["trial"] = float(trial)
            trial_rows.append(row)

            axbycz_result = solve_axbycz(axbycz_instance, dynamics=dynamics, projector=projector)
            axbycz_errors = [
                pose_error(axbycz_instance.x_true, axbycz_result.x_estimate),
                pose_error(axbycz_instance.y_true, axbycz_result.y_estimate),
                pose_error(axbycz_instance.z_true, axbycz_result.z_estimate),
            ]
            row = _record_problem_rows(
                "axbycz",
                method_name,
                max(error.rotation_deg for error in axbycz_errors),
                max(error.translation for error in axbycz_errors),
                axbycz_loss(
                    axbycz_instance.motions_a,
                    axbycz_instance.motions_b,
                    axbycz_instance.motions_c,
                    axbycz_result.x_estimate,
                    axbycz_result.y_estimate,
                    axbycz_result.z_estimate,
                ),
            )
            row["trial"] = float(trial)
            trial_rows.append(row)

    trial_records = pd.DataFrame(trial_rows)
    summary_by_problem = (
        trial_records.groupby(["problem", "method"], as_index=False)
        .agg(
            mean_rotation_error_deg=("rotation_error_deg", "mean"),
            mean_translation_error=("translation_error", "mean"),
            mean_loss=("loss", "mean"),
            success_rate_rot_le_5deg=("success_rot_le_5deg", "mean"),
        )
        .sort_values(["problem", "method"]) 
        .reset_index(drop=True)
    )
    overall_summary = (
        trial_records.groupby(["method"], as_index=False)
        .agg(
            mean_rotation_error_deg=("rotation_error_deg", "mean"),
            mean_translation_error=("translation_error", "mean"),
            mean_loss=("loss", "mean"),
            success_rate_rot_le_5deg=("success_rot_le_5deg", "mean"),
        )
        .sort_values(["method"]) 
        .reset_index(drop=True)
    )
    history_df = pd.DataFrame(histories)

    figures: Dict[str, str] = {}
    if output_dir is not None:
        trial_records.to_csv(f"{output_dir}/mc240_trial_records.csv", index=False)
        summary_by_problem.to_csv(f"{output_dir}/mc240_summary_by_problem.csv", index=False)
        overall_summary.to_csv(f"{output_dir}/mc240_overall_summary.csv", index=False)
        history_df.to_csv(f"{output_dir}/hand_eye_convergence_trial0.csv", index=False)

        plt.figure(figsize=(8, 4.6))
        for method_name in summary_by_problem["method"].unique():
            subset = summary_by_problem[summary_by_problem["method"] == method_name]
            plt.plot(subset["problem"], subset["mean_rotation_error_deg"], marker="o", label=method_name)
        plt.ylabel("Mean worst-case rotation error (deg)")
        plt.xlabel("Problem")
        plt.title("Corrected Zhang/CORO calibration benchmark")
        plt.legend()
        plt.tight_layout()
        rotation_plot_path = f"{output_dir}/rotation_summary.png"
        plt.savefig(rotation_plot_path, dpi=160)
        plt.close()
        figures["rotation_summary"] = rotation_plot_path

        plt.figure(figsize=(8, 4.6))
        for method_name in history_df["method"].unique():
            subset = history_df[(history_df["problem"] == "hand_eye") & (history_df["method"] == method_name)]
            plt.plot(subset["iteration"], subset["loss"], label=method_name)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Hand-eye loss")
        plt.title("Hand-eye convergence on trial 0")
        plt.legend()
        plt.tight_layout()
        convergence_plot_path = f"{output_dir}/hand_eye_convergence_trial0.png"
        plt.savefig(convergence_plot_path, dpi=160)
        plt.close()
        figures["hand_eye_convergence"] = convergence_plot_path

    return {
        "trial_records": trial_records,
        "summary_by_problem": summary_by_problem,
        "overall_summary": overall_summary,
        "history": history_df,
        "figures": pd.DataFrame([{key: value for key, value in figures.items()}]) if figures else pd.DataFrame(),
    }
