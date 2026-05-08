
"""CORO-only Monte Carlo benchmark for Zhang-style parallel calibration solvers.

This module keeps the corrected Zhang neural network formulation from the package
and evaluates only CORO-projected solvers:

- ZNN-CORO
- GNN-CORO

No SVD baseline is included. For each synthetic problem instance, the solver
history is stored so that all trial-wise convergence curves can be plotted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import (
    AXBYCZInstance,
    HandEyeInstance,
    RWHEInstance,
    axbycz_loss,
    generate_axbycz_instance,
    generate_hand_eye_instance,
    generate_rwhe_instance,
    rwhe_loss,
    hand_eye_loss,
    solve_axbycz,
    solve_hand_eye,
    solve_rwhe,
)
from .geometry import pose_error


METHOD_SPECS: Tuple[Tuple[str, str], ...] = (
    ("ZNN-CORO", "znn"),
    ("GNN-CORO", "gnn"),
)


@dataclass
class COROOnlyBenchmarkOutputs:
    hand_eye_trials: pd.DataFrame
    rwhe_trials: pd.DataFrame
    axbycz_trials: pd.DataFrame
    convergence_histories: pd.DataFrame
    summary_hand_eye: pd.DataFrame
    summary_rwhe: pd.DataFrame
    summary_axbycz: pd.DataFrame
    overall_summary: pd.DataFrame


def _make_summary(df: pd.DataFrame, metric_columns: Sequence[str], success_column: Optional[str] = None) -> pd.DataFrame:
    agg: Dict[str, Tuple[str, str]] = {}
    for metric in metric_columns:
        agg[f"mean_{metric}"] = (metric, "mean")
        agg[f"median_{metric}"] = (metric, "median")
        agg[f"std_{metric}"] = (metric, "std")
    if success_column is not None:
        agg[f"rate_{success_column}"] = (success_column, "mean")
    return (
        df.groupby("method", as_index=False)
        .agg(**agg)
        .sort_values("method")
        .reset_index(drop=True)
    )


def _all_trials_plot(
    history_df: pd.DataFrame,
    problem: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8.2, 4.9))
    for color_index, (method_name, _) in enumerate(METHOD_SPECS):
        subset = history_df[(history_df["problem"] == problem) & (history_df["method"] == method_name)]
        for _, trial_df in subset.groupby("trial"):
            plt.plot(
                trial_df["iteration"].to_numpy(),
                trial_df["loss"].to_numpy(),
                color=f"C{color_index}",
                alpha=0.08,
                linewidth=0.7,
            )
        median_curve = subset.groupby("iteration", as_index=False)["loss"].median()
        plt.plot(
            median_curve["iteration"].to_numpy(),
            median_curve["loss"].to_numpy(),
            color=f"C{color_index}",
            linewidth=2.5,
            label=f"{method_name} median",
        )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Calibration loss")
    plt.title(title if title is not None else problem.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=180)
    plt.close()


def _summary_bar_plot(
    summary_df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8.2, 4.6))
    labels = summary_df["problem_label"].tolist()
    method_names = [spec[0] for spec in METHOD_SPECS]
    x = np.arange(len(labels))
    width = 0.35
    for method_idx, method_name in enumerate(method_names):
        subset = summary_df[summary_df["method"] == method_name].set_index("problem_label").reindex(labels)
        positions = x + (method_idx - 0.5) * width
        plt.bar(positions, subset[value_column].to_numpy(), width=width, label=method_name)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=180)
    plt.close()


def benchmark_coro_only_trials(
    num_trials: int = 120,
    seed: int = 29,
    output_dir: Optional[str] = None,
) -> COROOnlyBenchmarkOutputs:
    """Run >100 CORO-only Monte Carlo trials for all three calibration problems."""
    rng = np.random.default_rng(seed)

    hand_rows: List[Dict[str, float]] = []
    rwhe_rows: List[Dict[str, float]] = []
    axbycz_rows: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []

    for trial in range(int(num_trials)):
        trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

        hand_instance = generate_hand_eye_instance(trial_rng)
        rwhe_instance = generate_rwhe_instance(trial_rng)
        axbycz_instance = generate_axbycz_instance(trial_rng)

        for method_name, dynamics in METHOD_SPECS:
            hand_result = solve_hand_eye(hand_instance, dynamics=dynamics, projector="coro")
            hand_err = pose_error(hand_instance.x_true, hand_result.estimate)
            hand_rows.append(
                {
                    "trial": float(trial),
                    "method": method_name,
                    "rotation_error_x_deg": hand_err.rotation_deg,
                    "translation_error_x": hand_err.translation,
                    "worst_rotation_error_deg": hand_err.rotation_deg,
                    "worst_translation_error": hand_err.translation,
                    "final_loss": hand_eye_loss(hand_instance.motions_a, hand_instance.motions_b, hand_result.estimate),
                    "success_rot_le_5deg": 1.0 if hand_err.rotation_deg <= 5.0 else 0.0,
                }
            )
            for item in hand_result.history:
                history_rows.append(
                    {
                        "problem": "hand_eye",
                        "trial": float(trial),
                        "method": method_name,
                        "iteration": item["iteration"],
                        "loss": item["loss"],
                        "residual_norm": item["residual_norm"],
                        "mass_condition": item["mass_condition"],
                    }
                )

            rwhe_result = solve_rwhe(rwhe_instance, dynamics=dynamics, projector="coro")
            rwhe_err_x = pose_error(rwhe_instance.x_true, rwhe_result.x_estimate)
            rwhe_err_y = pose_error(rwhe_instance.y_true, rwhe_result.y_estimate)
            rwhe_rows.append(
                {
                    "trial": float(trial),
                    "method": method_name,
                    "rotation_error_x_deg": rwhe_err_x.rotation_deg,
                    "translation_error_x": rwhe_err_x.translation,
                    "rotation_error_y_deg": rwhe_err_y.rotation_deg,
                    "translation_error_y": rwhe_err_y.translation,
                    "worst_rotation_error_deg": max(rwhe_err_x.rotation_deg, rwhe_err_y.rotation_deg),
                    "worst_translation_error": max(rwhe_err_x.translation, rwhe_err_y.translation),
                    "final_loss": rwhe_loss(rwhe_instance.motions_a, rwhe_instance.motions_b, rwhe_result.x_estimate, rwhe_result.y_estimate),
                    "success_rot_le_5deg": 1.0 if max(rwhe_err_x.rotation_deg, rwhe_err_y.rotation_deg) <= 5.0 else 0.0,
                }
            )
            for item in rwhe_result.history:
                history_rows.append(
                    {
                        "problem": "rwhe",
                        "trial": float(trial),
                        "method": method_name,
                        "iteration": item["iteration"],
                        "loss": item["loss"],
                        "residual_norm": item["residual_norm"],
                        "mass_condition": item["mass_condition"],
                    }
                )

            axbycz_result = solve_axbycz(axbycz_instance, dynamics=dynamics, projector="coro")
            ax_err_x = pose_error(axbycz_instance.x_true, axbycz_result.x_estimate)
            ax_err_y = pose_error(axbycz_instance.y_true, axbycz_result.y_estimate)
            ax_err_z = pose_error(axbycz_instance.z_true, axbycz_result.z_estimate)
            axbycz_rows.append(
                {
                    "trial": float(trial),
                    "method": method_name,
                    "rotation_error_x_deg": ax_err_x.rotation_deg,
                    "translation_error_x": ax_err_x.translation,
                    "rotation_error_y_deg": ax_err_y.rotation_deg,
                    "translation_error_y": ax_err_y.translation,
                    "rotation_error_z_deg": ax_err_z.rotation_deg,
                    "translation_error_z": ax_err_z.translation,
                    "worst_rotation_error_deg": max(ax_err_x.rotation_deg, ax_err_y.rotation_deg, ax_err_z.rotation_deg),
                    "worst_translation_error": max(ax_err_x.translation, ax_err_y.translation, ax_err_z.translation),
                    "final_loss": axbycz_loss(
                        axbycz_instance.motions_a,
                        axbycz_instance.motions_b,
                        axbycz_instance.motions_c,
                        axbycz_result.x_estimate,
                        axbycz_result.y_estimate,
                        axbycz_result.z_estimate,
                    ),
                    "success_rot_le_5deg": 1.0 if max(ax_err_x.rotation_deg, ax_err_y.rotation_deg, ax_err_z.rotation_deg) <= 5.0 else 0.0,
                }
            )
            for item in axbycz_result.history:
                history_rows.append(
                    {
                        "problem": "axbycz",
                        "trial": float(trial),
                        "method": method_name,
                        "iteration": item["iteration"],
                        "loss": item["loss"],
                        "residual_norm": item["residual_norm"],
                        "mass_condition": item["mass_condition"],
                    }
                )

    hand_df = pd.DataFrame(hand_rows)
    rwhe_df = pd.DataFrame(rwhe_rows)
    axbycz_df = pd.DataFrame(axbycz_rows)
    history_df = pd.DataFrame(history_rows)

    summary_hand = _make_summary(
        hand_df,
        metric_columns=[
            "rotation_error_x_deg",
            "translation_error_x",
            "worst_rotation_error_deg",
            "worst_translation_error",
            "final_loss",
        ],
        success_column="success_rot_le_5deg",
    )
    summary_rwhe = _make_summary(
        rwhe_df,
        metric_columns=[
            "rotation_error_x_deg",
            "translation_error_x",
            "rotation_error_y_deg",
            "translation_error_y",
            "worst_rotation_error_deg",
            "worst_translation_error",
            "final_loss",
        ],
        success_column="success_rot_le_5deg",
    )
    summary_axbycz = _make_summary(
        axbycz_df,
        metric_columns=[
            "rotation_error_x_deg",
            "translation_error_x",
            "rotation_error_y_deg",
            "translation_error_y",
            "rotation_error_z_deg",
            "translation_error_z",
            "worst_rotation_error_deg",
            "worst_translation_error",
            "final_loss",
        ],
        success_column="success_rot_le_5deg",
    )

    overall_rows: List[Dict[str, float]] = []
    for problem_name, df in (
        ("Hand-eye", hand_df),
        ("Robot-world / hand-eye", rwhe_df),
        ("Hand-eye / robot-world / tool-flange", axbycz_df),
    ):
        grouped = (
            df.groupby("method", as_index=False)
            .agg(
                mean_worst_rotation_error_deg=("worst_rotation_error_deg", "mean"),
                median_worst_rotation_error_deg=("worst_rotation_error_deg", "median"),
                std_worst_rotation_error_deg=("worst_rotation_error_deg", "std"),
                mean_worst_translation_error=("worst_translation_error", "mean"),
                median_worst_translation_error=("worst_translation_error", "median"),
                std_worst_translation_error=("worst_translation_error", "std"),
                mean_final_loss=("final_loss", "mean"),
                median_final_loss=("final_loss", "median"),
                rate_success_rot_le_5deg=("success_rot_le_5deg", "mean"),
            )
            .sort_values("method")
        )
        grouped["problem"] = problem_name
        overall_rows.append(grouped)
    overall_summary = pd.concat(overall_rows, axis=0, ignore_index=True)
    overall_summary = overall_summary[[
        "problem",
        "method",
        "mean_worst_rotation_error_deg",
        "median_worst_rotation_error_deg",
        "std_worst_rotation_error_deg",
        "mean_worst_translation_error",
        "median_worst_translation_error",
        "std_worst_translation_error",
        "mean_final_loss",
        "median_final_loss",
        "rate_success_rot_le_5deg",
    ]]

    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        out = pd.Path if False else None  # no-op to keep linters quiet
        hand_df.to_csv(f"{output_dir}/hand_eye_trials.csv", index=False)
        rwhe_df.to_csv(f"{output_dir}/rwhe_trials.csv", index=False)
        axbycz_df.to_csv(f"{output_dir}/axbycz_trials.csv", index=False)
        history_df.to_csv(f"{output_dir}/convergence_histories.csv", index=False)
        summary_hand.to_csv(f"{output_dir}/summary_hand_eye.csv", index=False)
        summary_rwhe.to_csv(f"{output_dir}/summary_rwhe.csv", index=False)
        summary_axbycz.to_csv(f"{output_dir}/summary_axbycz.csv", index=False)
        overall_summary.to_csv(f"{output_dir}/overall_summary.csv", index=False)

        _all_trials_plot(
            history_df,
            problem="hand_eye",
            output_path=f"{output_dir}/hand_eye_convergence_all_trials.png",
            title="Hand-eye: all trial convergence curves",
        )
        _all_trials_plot(
            history_df,
            problem="rwhe",
            output_path=f"{output_dir}/rwhe_convergence_all_trials.png",
            title="Robot-world / hand-eye: all trial convergence curves",
        )
        _all_trials_plot(
            history_df,
            problem="axbycz",
            output_path=f"{output_dir}/axbycz_convergence_all_trials.png",
            title="Hand-eye / robot-world / tool-flange: all trial convergence curves",
        )

        rotation_summary_plot = pd.concat(
            [
                summary_hand[["method", "mean_worst_rotation_error_deg"]].assign(problem_label="Hand-eye"),
                summary_rwhe[["method", "mean_worst_rotation_error_deg"]].assign(problem_label="RWHE"),
                summary_axbycz[["method", "mean_worst_rotation_error_deg"]].assign(problem_label="AXB=YCZ"),
            ],
            axis=0,
            ignore_index=True,
        )
        _summary_bar_plot(
            rotation_summary_plot,
            value_column="mean_worst_rotation_error_deg",
            ylabel="Mean worst-case rotation error (deg)",
            title="CORO-only 120-trial benchmark",
            output_path=f"{output_dir}/rotation_error_summary.png",
        )

        translation_summary_plot = pd.concat(
            [
                summary_hand[["method", "mean_worst_translation_error"]].assign(problem_label="Hand-eye"),
                summary_rwhe[["method", "mean_worst_translation_error"]].assign(problem_label="RWHE"),
                summary_axbycz[["method", "mean_worst_translation_error"]].assign(problem_label="AXB=YCZ"),
            ],
            axis=0,
            ignore_index=True,
        )
        _summary_bar_plot(
            translation_summary_plot,
            value_column="mean_worst_translation_error",
            ylabel="Mean worst-case translation error",
            title="CORO-only 120-trial benchmark",
            output_path=f"{output_dir}/translation_error_summary.png",
        )

    return COROOnlyBenchmarkOutputs(
        hand_eye_trials=hand_df,
        rwhe_trials=rwhe_df,
        axbycz_trials=axbycz_df,
        convergence_histories=history_df,
        summary_hand_eye=summary_hand,
        summary_rwhe=summary_rwhe,
        summary_axbycz=summary_axbycz,
        overall_summary=overall_summary,
    )
