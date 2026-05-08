from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_coro_opencl.reference import (
    deep_coro_forward_np,
    make_distorted_rotations,
    orthogonality_error,
    rotation_error_deg,
    svd_project_so3,
)
from deep_coro_opencl.opencl_operator import DeepCoroOpenCLOperator, OpenCLUnavailableError


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })


def time_function(fn, repeats: int = 5, warmup: int = 2) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


def convergence_experiment(out_dir: Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    B = 2048
    M, R_gt = make_distorted_rotations(B, rng, dtype=np.float32)
    R_svd = svd_project_so3(M)
    layers_list = [1, 2, 4, 6, 8, 12, 16]
    rows: List[Dict[str, float]] = []
    for L in layers_list:
        alpha = np.ones(L, dtype=np.float32)
        beta = np.ones(L, dtype=np.float32)
        R = deep_coro_forward_np(M, alpha, beta)
        orth = orthogonality_error(R)
        det_err = np.abs(np.linalg.det(R.astype(np.float64)) - 1.0)
        angle_gt = rotation_error_deg(R, R_gt)
        angle_svd = rotation_error_deg(R, R_svd)
        rows.append({
            "layers": L,
            "mean_orthogonality_error": float(np.mean(orth)),
            "median_orthogonality_error": float(np.median(orth)),
            "p90_orthogonality_error": float(np.quantile(orth, 0.90)),
            "mean_det_error": float(np.mean(det_err)),
            "mean_angle_to_gt_deg": float(np.mean(angle_gt)),
            "median_angle_to_gt_deg": float(np.median(angle_gt)),
            "mean_angle_to_svd_deg": float(np.mean(angle_svd)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "coro_convergence_vs_layers.csv", index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    ax.semilogy(df["layers"], df["mean_orthogonality_error"], marker="o", linewidth=2.6, label="Mean")
    ax.semilogy(df["layers"], df["p90_orthogonality_error"], marker="s", linewidth=2.2, label="90th percentile")
    ax.set_xlabel("Deep CORO layers")
    ax.set_ylabel(r"$\|\mathbf{R}^{\top}\mathbf{R}-\mathbf{I}\|_{F}$")
    ax.set_title("Deep CORO convergence under the OpenCL kernel map")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "coro_convergence_vs_layers.png", dpi=300)
    fig.savefig(out_dir / "coro_convergence_vs_layers.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    ax.plot(df["layers"], df["mean_angle_to_gt_deg"], marker="o", linewidth=2.6, label="To ground truth")
    ax.plot(df["layers"], df["mean_angle_to_svd_deg"], marker="s", linewidth=2.2, label="To SVD projection")
    ax.set_xlabel("Deep CORO layers")
    ax.set_ylabel("Mean rotation error (deg)")
    ax.set_title("Accuracy after Deep CORO projection")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "coro_angle_vs_layers.png", dpi=300)
    fig.savefig(out_dir / "coro_angle_vs_layers.svg")
    plt.close(fig)
    return df


def try_create_opencl(fast_math: bool = False):
    try:
        return DeepCoroOpenCLOperator(fast_math=fast_math, profiling=True)
    except Exception as exc:
        return exc


def runtime_experiment(out_dir: Path, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    batch_sizes = [32, 128, 512, 2048, 8192]
    L = 8
    rows = []
    alpha = np.ones(L, dtype=np.float32)
    beta = np.ones(L, dtype=np.float32)

    # CPU vectorized NumPy reference and SVD baseline.
    for B in batch_sizes:
        M, _ = make_distorted_rotations(B, rng, dtype=np.float32)
        t_coro = time_function(lambda: deep_coro_forward_np(M, alpha, beta), repeats=5, warmup=2)
        t_svd = time_function(lambda: svd_project_so3(M), repeats=3, warmup=1)
        rows.append({
            "backend": "numpy_cpu",
            "operator": "deep_coro_reference",
            "batch": B,
            "layers": L,
            "median_time_ms": float(np.median(t_coro) * 1e3),
            "mean_time_ms": float(np.mean(t_coro) * 1e3),
            "median_us_per_matrix": float(np.median(t_coro) * 1e6 / B),
            "available": 1,
        })
        rows.append({
            "backend": "numpy_cpu",
            "operator": "svd_projection",
            "batch": B,
            "layers": 0,
            "median_time_ms": float(np.median(t_svd) * 1e3),
            "mean_time_ms": float(np.mean(t_svd) * 1e3),
            "median_us_per_matrix": float(np.median(t_svd) * 1e6 / B),
            "available": 1,
        })

    # Optional OpenCL benchmark.  This path is skipped cleanly if no device exists.
    op = try_create_opencl(fast_math=False)
    if isinstance(op, DeepCoroOpenCLOperator):
        info = op.info()
        for B in batch_sizes:
            M, _ = make_distorted_rotations(B, rng, dtype=np.float32)
            # One warmup plus profiled repeats.
            op.forward(M, alpha, beta)
            times = []
            for _ in range(10):
                _, event_ms = op.forward(M, alpha, beta, return_event_time=True)
                if event_ms is not None:
                    times.append(event_ms)
            if times:
                t = np.asarray(times) / 1e3  # seconds
                rows.append({
                    "backend": f"opencl:{info.get('device', 'device')}",
                    "operator": "opencl_deep_coro",
                    "batch": B,
                    "layers": L,
                    "median_time_ms": float(np.median(t) * 1e3),
                    "mean_time_ms": float(np.mean(t) * 1e3),
                    "median_us_per_matrix": float(np.median(t) * 1e6 / B),
                    "available": 1,
                })
    else:
        rows.append({
            "backend": "opencl",
            "operator": "opencl_deep_coro",
            "batch": 0,
            "layers": L,
            "median_time_ms": np.nan,
            "mean_time_ms": np.nan,
            "median_us_per_matrix": np.nan,
            "available": 0,
            "error": str(op),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operator_runtime.csv", index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=220)
    valid = df[(df["available"] == 1) & (df["batch"] > 0)]
    for (backend, operator), group in valid.groupby(["backend", "operator"]):
        label = f"{operator.replace('_', ' ')} ({backend})"
        ax.loglog(group["batch"], group["median_us_per_matrix"], marker="o", linewidth=2.4, label=label)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(r"Median runtime per matrix ($\mu$s)")
    ax.set_title("Deep CORO OpenCL operator runtime")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True, loc="best")
    if not (df["operator"].eq("opencl_deep_coro") & df["available"].eq(1)).any():
        ax.text(0.04, 0.04, "OpenCL device not available in this environment", transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_dir / "operator_runtime.png", dpi=300)
    fig.savefig(out_dir / "operator_runtime.svg")
    plt.close(fig)
    return df


def consistency_experiment(out_dir: Path, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    B = 1024
    L = 8
    M, _ = make_distorted_rotations(B, rng, dtype=np.float32)
    alpha = np.ones(L, dtype=np.float32)
    beta = np.ones(L, dtype=np.float32)
    R_ref = deep_coro_forward_np(M, alpha, beta)
    rows = []
    op = try_create_opencl(fast_math=False)
    if isinstance(op, DeepCoroOpenCLOperator):
        R_opencl = op.forward(M, alpha, beta)
        rows.append({
            "path": "opencl_f32_vs_numpy_reference",
            "available": 1,
            "max_abs_difference": float(np.max(np.abs(R_opencl - R_ref))),
            "mean_abs_difference": float(np.mean(np.abs(R_opencl - R_ref))),
            **{f"device_{k}": v for k, v in op.info().items() if isinstance(v, (str, int, float))},
        })
    else:
        rows.append({
            "path": "opencl_f32_vs_numpy_reference",
            "available": 0,
            "max_abs_difference": np.nan,
            "mean_abs_difference": np.nan,
            "error": str(op),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operator_consistency.csv", index=False)
    return df


def write_summary(out_dir: Path, conv: pd.DataFrame, runtime: pd.DataFrame, consistency: pd.DataFrame) -> None:
    md = []
    md.append("# Deep CORO OpenCL operator summary\n")
    md.append("## Convergence versus layer count\n")
    md.append(conv[["layers", "mean_orthogonality_error", "p90_orthogonality_error", "mean_angle_to_svd_deg"]].to_markdown(index=False))
    md.append("\n\n## Runtime\n")
    md.append(runtime[["backend", "operator", "batch", "layers", "median_time_ms", "median_us_per_matrix", "available"]].to_markdown(index=False))
    md.append("\n\n## Consistency\n")
    md.append(consistency.to_markdown(index=False))
    (out_dir / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Deep CORO OpenCL/reference experiments.")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conv = convergence_experiment(args.out)
    runtime = runtime_experiment(args.out)
    consistency = consistency_experiment(args.out)
    write_summary(args.out, conv, runtime, consistency)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
