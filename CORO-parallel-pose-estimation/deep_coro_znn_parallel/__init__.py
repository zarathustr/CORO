"""Corrected Zhang-neural-network + CORO parallel calibration package."""

from .activations import matrix_activation
from .coro import project_so, project_se3_rotation
from .calibration import (
    benchmark_calibration_suite,
    generate_axbycz_instance,
    generate_hand_eye_instance,
    generate_rwhe_instance,
    solve_axbycz,
    solve_hand_eye,
    solve_rwhe,
)
from .zhang_reference import run_zhang_reference_benchmark
from .coro_only_trials import benchmark_coro_only_trials

__all__ = [
    "matrix_activation",
    "project_so",
    "project_se3_rotation",
    "benchmark_calibration_suite",
    "generate_axbycz_instance",
    "generate_hand_eye_instance",
    "generate_rwhe_instance",
    "solve_axbycz",
    "solve_hand_eye",
    "solve_rwhe",
    "run_zhang_reference_benchmark",
    "benchmark_coro_only_trials",
]
