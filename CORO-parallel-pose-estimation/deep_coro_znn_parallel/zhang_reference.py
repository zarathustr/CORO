"""Direct reference benchmark based on Zhang, Fan, Li (2007)."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .activations import matrix_activation
from .geometry import vec_f, mat_f



def _matrix_a(time_value: float) -> np.ndarray:
    return np.array(
        [
            [np.sin(time_value), np.cos(time_value)],
            [-np.cos(time_value), np.sin(time_value)],
        ],
        dtype=float,
    )



def _matrix_a_dot(time_value: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(time_value), -np.sin(time_value)],
            [np.sin(time_value), np.cos(time_value)],
        ],
        dtype=float,
    )



def _matrix_b(time_value: float) -> np.ndarray:
    _ = time_value
    return np.zeros((2, 2), dtype=float)



def _matrix_b_dot(time_value: float) -> np.ndarray:
    _ = time_value
    return np.zeros((2, 2), dtype=float)



def _matrix_c(time_value: float) -> np.ndarray:
    _ = time_value
    return -np.eye(2, dtype=float)



def _matrix_c_dot(time_value: float) -> np.ndarray:
    _ = time_value
    return np.zeros((2, 2), dtype=float)



def _theoretical_solution(time_value: float) -> np.ndarray:
    return np.linalg.inv(_matrix_a(time_value))



def _znn_rhs(time_value: float, state: np.ndarray, gamma: float, activation: str) -> np.ndarray:
    matrix_a = _matrix_a(time_value)
    matrix_b = _matrix_b(time_value)
    matrix_c = _matrix_c(time_value)
    matrix_a_dot = _matrix_a_dot(time_value)
    matrix_b_dot = _matrix_b_dot(time_value)
    matrix_c_dot = _matrix_c_dot(time_value)
    error = matrix_a @ state - state @ matrix_b + matrix_c
    rhs = -matrix_a_dot @ state + state @ matrix_b_dot - matrix_c_dot - gamma * matrix_activation(error, kind=activation)
    if np.linalg.norm(matrix_b) < 1e-12:
        derivative = np.linalg.solve(matrix_a, rhs)
    else:
        operator = np.kron(np.eye(2), matrix_a) - np.kron(matrix_b.T, np.eye(2))
        derivative = mat_f(np.linalg.lstsq(operator, vec_f(rhs), rcond=None)[0], 2, 2)
    return derivative



def _gnn_rhs(time_value: float, state: np.ndarray, gamma: float, activation: str) -> np.ndarray:
    matrix_a = _matrix_a(time_value)
    matrix_b = _matrix_b(time_value)
    matrix_c = _matrix_c(time_value)
    error = matrix_a @ state - state @ matrix_b + matrix_c
    activated = matrix_activation(error, kind=activation)
    return -gamma * (matrix_a.T @ activated - activated @ matrix_b.T)



def run_zhang_reference_benchmark(output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Run the 2 x 2 time-varying Sylvester benchmark from Zhang 2007."""
    horizon = 10.0
    dt = 0.01
    time_grid = np.arange(0.0, horizon + 0.5 * dt, dt)
    methods = [
        ("ZNN-linear", "znn", "linear", 1.0),
        ("ZNN-power-sigmoid", "znn", "power_sigmoid", 1.0),
        ("GNN-linear", "gnn", "linear", 1.0),
        ("GNN-power-sigmoid", "gnn", "power_sigmoid", 1.0),
    ]
    initial_states = [
        np.array([[1.4, -1.3], [0.7, 1.8]], dtype=float),
        np.array([[-1.1, 0.9], [1.6, -0.8]], dtype=float),
        np.array([[0.8, 1.5], [-1.2, 0.6]], dtype=float),
    ]
    trial_rows: List[Dict[str, float]] = []
    trajectory_rows: List[Dict[str, float]] = []
    for trial_index, initial_state in enumerate(initial_states):
        for method_name, dynamics, activation, gamma in methods:
            state = initial_state.copy()
            for time_value in time_grid:
                theory = _theoretical_solution(time_value)
                error_norm = float(np.linalg.norm(state - theory))
                trajectory_rows.append(
                    {
                        "trial": float(trial_index),
                        "method": method_name,
                        "time": float(time_value),
                        "tracking_error_fro": error_norm,
                        "x11": float(state[0, 0]),
                        "x12": float(state[0, 1]),
                        "x21": float(state[1, 0]),
                        "x22": float(state[1, 1]),
                        "x11_true": float(theory[0, 0]),
                        "x12_true": float(theory[0, 1]),
                        "x21_true": float(theory[1, 0]),
                        "x22_true": float(theory[1, 1]),
                    }
                )
                derivative = _znn_rhs(time_value, state, gamma, activation) if dynamics == "znn" else _gnn_rhs(time_value, state, gamma, activation)
                state = state + dt * derivative
            final_theory = _theoretical_solution(time_grid[-1])
            trial_rows.append(
                {
                    "trial": float(trial_index),
                    "method": method_name,
                    "final_tracking_error_fro": float(np.linalg.norm(state - final_theory)),
                    "mean_tracking_error_fro": float(np.mean([row["tracking_error_fro"] for row in trajectory_rows if row["trial"] == float(trial_index) and row["method"] == method_name])),
                }
            )
    trial_df = pd.DataFrame(trial_rows)
    summary_df = trial_df.groupby(["method"], as_index=False).agg(
        mean_final_tracking_error_fro=("final_tracking_error_fro", "mean"),
        mean_tracking_error_fro=("mean_tracking_error_fro", "mean"),
    )
    trajectory_df = pd.DataFrame(trajectory_rows)

    if output_dir is not None:
        trial_df.to_csv(f"{output_dir}/zhang_reference_trials.csv", index=False)
        summary_df.to_csv(f"{output_dir}/zhang_reference_summary.csv", index=False)
        trajectory_df.to_csv(f"{output_dir}/zhang_reference_trajectories.csv", index=False)

        plt.figure(figsize=(8.0, 4.5))
        for method_name in summary_df["method"]:
            subset = trajectory_df[(trajectory_df["trial"] == 0.0) & (trajectory_df["method"] == method_name)]
            plt.plot(subset["time"], subset["tracking_error_fro"], label=method_name)
        plt.yscale("log")
        plt.xlabel("Time (s)")
        plt.ylabel("Tracking error Frobenius norm")
        plt.title("Zhang 2007 reference benchmark")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/zhang_reference_tracking_error.png", dpi=160)
        plt.close()

    return {
        "trials": trial_df,
        "summary": summary_df,
        "trajectories": trajectory_df,
    }
