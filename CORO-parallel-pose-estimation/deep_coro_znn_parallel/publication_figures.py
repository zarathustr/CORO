"""Publication figure regeneration helpers for the parallel calibration study.

This module regenerates the manuscript-facing figures requested by the user:

- lifted_benchmark_summary.png
- direct_error_characteristics.png
- hand_eye_convergence_all_trials.png
- rwhe_convergence_all_trials.png
- axbycz_convergence_all_trials.png

The direct-error and convergence figures are regenerated from the saved MC120
trial records. The lifted benchmark summary is regenerated from the companion
SO(4)-lifting summary statistics bundled with this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROBLEM_ABBREVIATIONS: Dict[str, str] = {
    "Hand-eye": "HEC",
    "Robot-world / hand-eye": "HERW",
    "Hand-eye / robot-world / tool-flange": "HERWTF",
}

PROBLEM_HISTORY_KEYS: Dict[str, str] = {
    "hand_eye": "HEC",
    "rwhe": "HERW",
    "axbycz": "HERWTF",
}

DIRECT_METHOD_ORDER: Sequence[str] = (
    "ZNN-CORO",
    "GNN-CORO",
)

LIFTED_METHOD_ORDER: Sequence[str] = (
    "ZNN-CORO",
    "SE(3)++ Lift + ZNN-CORO",
    "Cartan Lift + ZNN-CORO",
    "GNN-CORO",
)

METHOD_COLORS: Dict[str, str] = {
    "ZNN-CORO": "#1f77b4",
    "GNN-CORO": "#d62728",
    "SE(3)++ Lift + ZNN-CORO": "#2ca02c",
    "Cartan Lift + ZNN-CORO": "#9467bd",
}

METHOD_MARKERS: Dict[str, str] = {
    "ZNN-CORO": "o",
    "GNN-CORO": "s",
    "SE(3)++ Lift + ZNN-CORO": "^",
    "Cartan Lift + ZNN-CORO": "D",
}

METHOD_LINESTYLES: Dict[str, str] = {
    "ZNN-CORO": "-",
    "GNN-CORO": "--",
}


def _installed_fonts() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def _preferred_serif_stack(prefer_times: bool) -> List[str]:
    if prefer_times:
        return ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
    return ["Nimbus Roman", "Liberation Serif", "DejaVu Serif", "Times New Roman"]


def _active_primary_font(prefer_times: bool) -> str:
    installed = _installed_fonts()
    for name in _preferred_serif_stack(prefer_times):
        if name in installed:
            return name
    return "DejaVu Serif"


def _figure_style(font_size: int = 18, prefer_times: bool = False) -> Dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": _preferred_serif_stack(prefer_times),
        "mathtext.fontset": "stix",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "figure.titlesize": font_size + 2,
        "axes.linewidth": 3.0,
        "lines.linewidth": 4.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }


def _load_direct_summary(project_root: Path) -> pd.DataFrame:
    overall = pd.read_csv(project_root / "results" / "coro_only_mc120" / "overall_summary.csv")
    overall = overall.copy()
    overall["problem_abbr"] = overall["problem"].map(PROBLEM_ABBREVIATIONS)
    return overall


def _load_histories(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "results" / "coro_only_mc120" / "convergence_histories.csv")


def _load_lifted_summary(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "results" / "lifted_companion" / "lifted_summary.csv")


def plot_direct_error_characteristics(project_root: Path, output_path: Path) -> None:
    summary = _load_direct_summary(project_root)
    problem_order = ["HEC", "HERW", "HERWTF"]

    with plt.rc_context(_figure_style(font_size=18, prefer_times=False)):
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=False)
        x = np.arange(len(problem_order), dtype=float)
        offsets = np.array([-0.08, 0.08])

        panels = [
            (
                axes[0],
                "mean_worst_rotation_error_deg",
                "std_worst_rotation_error_deg",
                r"Worst-case Rotation Error ($^\circ$)",
                "Rotation",
            ),
            (
                axes[1],
                "mean_worst_translation_error",
                "std_worst_translation_error",
                "Worst-case Translation Error",
                "Translation",
            ),
        ]

        for ax, mean_col, std_col, ylabel, title in panels:
            for method_index, method in enumerate(DIRECT_METHOD_ORDER):
                subset = (
                    summary[summary["method"] == method]
                    .set_index("problem_abbr")
                    .reindex(problem_order)
                    .reset_index()
                )
                ax.errorbar(
                    x + offsets[method_index],
                    subset[mean_col].to_numpy(),
                    yerr=subset[std_col].to_numpy(),
                    fmt=METHOD_MARKERS[method] + METHOD_LINESTYLES[method],
                    color=METHOD_COLORS[method],
                    markersize=9,
                    linewidth=2.6,
                    capsize=5,
                    capthick=1.2,
                    label=method,
                    zorder=4,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(problem_order)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.28)
            ax.set_axisbelow(True)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            frameon=False,
            handlelength=2.4,
        )
        fig.subplots_adjust(top=0.80, wspace=0.28)
        fig.savefig(output_path, dpi=600)
        plt.close(fig)


def plot_lifted_benchmark_summary(project_root: Path, output_path: Path) -> str:
    summary = _load_lifted_summary(project_root)
    problem_order = ["HEC", "HERW", "HERWTF"]
    font_name = _active_primary_font(prefer_times=True)

    with plt.rc_context(_figure_style(font_size=20, prefer_times=True)):
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=False)
        x = np.arange(len(problem_order), dtype=float)
        width = 0.19
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(LIFTED_METHOD_ORDER))

        panel_specs = [
            (axes[0], "mean_worst_rotation_error_deg", r"Mean Worst-Case Rotation Error ($^\circ$)", (0.0, 14.0)),
            (axes[1], "success_rate_percent", "Success Rate (%)", (0.0, 110.0)),
        ]

        for ax, value_col, ylabel, ylim in panel_specs:
            for idx, method in enumerate(LIFTED_METHOD_ORDER):
                subset = (
                    summary[summary["method"] == method]
                    .set_index("problem")
                    .reindex(problem_order)
                    .reset_index()
                )
                ax.bar(
                    x + offsets[idx],
                    subset[value_col].to_numpy(),
                    width=width,
                    color=METHOD_COLORS[method],
                    edgecolor="black",
                    linewidth=0.6,
                    label=method,
                    zorder=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(problem_order)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)
            ax.set_axisbelow(True)

        axes[0].set_title("Rotation Performance")
        axes[1].set_title("Success Rate")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            ncol=2,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.0,
        )
        fig.subplots_adjust(top=0.76, wspace=0.24)
        fig.savefig(output_path, dpi=600)
        plt.close(fig)
    return font_name


def plot_convergence_all_trials(
    project_root: Path,
    problem_key: str,
    output_path: Path,
    title: str,
) -> None:
    histories = _load_histories(project_root)
    subset = histories[histories["problem"] == problem_key].copy()
    if subset.empty:
        raise ValueError(f"No history rows found for problem key: {problem_key}")

    trials = np.sort(subset["trial"].unique())
    trial_colors = plt.cm.prism(np.linspace(0.03, 0.97, len(trials)))

    with plt.rc_context(_figure_style(font_size=18, prefer_times=False)):
        fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=False)
        trial_style_handles = []
        for color, trial in zip(trial_colors, trials):
            trial_subset = subset[subset["trial"] == trial]
            for method in DIRECT_METHOD_ORDER:
                method_subset = trial_subset[trial_subset["method"] == method]
                if method_subset.empty:
                    continue
                ax.plot(
                    method_subset["iteration"].to_numpy(),
                    method_subset["loss"].to_numpy(),
                    color=color,
                    linestyle=METHOD_LINESTYLES[method],
                    alpha=1.0,
                    linewidth=3.0,
                    zorder=1,
                )

        mean_handles = []
        for method in DIRECT_METHOD_ORDER:
            mean_curve = (
                subset[subset["method"] == method]
                .groupby("iteration", as_index=False)["loss"]
                .mean()
            )
            handle, = ax.plot(
                mean_curve["iteration"].to_numpy(),
                mean_curve["loss"].to_numpy(),
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=10,
                label=f"{method} Mean",
                zorder=5,
            )
            mean_handles.append(handle)

        trial_style_handles = [
            Line2D([0], [0], color="0.35", linestyle="-", linewidth=3.8, label="ZNN-CORO Trials"),
            Line2D([0], [0], color="0.35", linestyle="--", linewidth=3.8, label="GNN-CORO Trials"),
        ]

        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Calibration Loss")
        # Title intentionally omitted so that the external legend never occludes the curves.
        ax.grid(True, which="major", axis="both", linewidth=1.6, alpha=0.28)
        ax.set_axisbelow(True)
        ax.legend(
            handles=trial_style_handles + mean_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.3,
        )
        fig.subplots_adjust(bottom=0.28, top=0.96)
        fig.savefig(output_path, dpi=600)
        plt.close(fig)


def regenerate_requested_figures(project_root: Path | None = None) -> Dict[str, str]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    plot_direct_error_characteristics(root, figures_dir / "direct_error_characteristics.png")
    outputs["direct_error_characteristics"] = str(figures_dir / "direct_error_characteristics.png")

    font_name = plot_lifted_benchmark_summary(root, figures_dir / "lifted_benchmark_summary.png")
    outputs["lifted_benchmark_summary"] = str(figures_dir / "lifted_benchmark_summary.png")
    outputs["lifted_benchmark_primary_font"] = font_name

    convergence_specs = [
        ("hand_eye", "hand_eye_convergence_all_trials.png", "HEC: all 120 trial convergence curves"),
        ("rwhe", "rwhe_convergence_all_trials.png", "HERW: all 120 trial convergence curves"),
        ("axbycz", "axbycz_convergence_all_trials.png", "HERWTF: all 120 trial convergence curves"),
    ]
    for problem_key, filename, title in convergence_specs:
        plot_convergence_all_trials(root, problem_key, figures_dir / filename, title)
        outputs[problem_key] = str(figures_dir / filename)

    return outputs

