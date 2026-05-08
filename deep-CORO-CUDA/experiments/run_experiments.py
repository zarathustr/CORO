from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

from deep_coro_cuda.ops import (
    deep_coro_forward,
    deep_coro_forward_torch,
    random_rotations,
    rotation_error_deg,
    svd_project_so3,
)


def make_distorted_rotations(n: int, dtype: torch.dtype, device: torch.device, seed: int = 0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # QR random rotations with global torch seed for compatibility
    torch.manual_seed(seed)
    R = random_rotations(n, dtype=dtype, device=device)
    log_scales = 0.60 * torch.randn(n, 3, dtype=dtype, device=device)
    scales = torch.exp(log_scales)
    noise = 0.025 * torch.randn(n, 3, 3, dtype=dtype, device=device)
    M = R @ torch.diag_embed(scales) + noise
    return M, R


def orthogonality_error(R: torch.Tensor) -> torch.Tensor:
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    return torch.linalg.norm(R.transpose(-1, -2) @ R - I, dim=(-2, -1))


def convergence_experiment(out_dir: Path) -> pd.DataFrame:
    device = torch.device("cpu")
    dtype = torch.float32
    N = 1536
    layers_list = [1, 2, 4, 6, 8, 12, 16]
    M, R_gt = make_distorted_rotations(N, dtype=dtype, device=device, seed=42)
    R_svd = svd_project_so3(M)
    rows: List[Dict[str, float]] = []
    for L in layers_list:
        alpha = torch.ones(L, dtype=dtype, device=device)
        beta = torch.ones(L, dtype=dtype, device=device)
        R = deep_coro_forward_torch(M, alpha, beta)
        orth = orthogonality_error(R)
        det_err = (torch.linalg.det(R) - 1.0).abs()
        angle_gt = rotation_error_deg(R, R_gt)
        angle_svd = rotation_error_deg(R, R_svd)
        rows.append(
            {
                "layers": L,
                "mean_orthogonality_error": float(orth.mean().item()),
                "median_orthogonality_error": float(orth.median().item()),
                "p90_orthogonality_error": float(torch.quantile(orth, 0.90).item()),
                "mean_det_error": float(det_err.mean().item()),
                "mean_angle_to_gt_deg": float(angle_gt.mean().item()),
                "median_angle_to_gt_deg": float(angle_gt.median().item()),
                "mean_angle_to_svd_deg": float(angle_svd.mean().item()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "coro_convergence_vs_layers.csv", index=False)

    plt.figure(figsize=(7.5, 4.8), dpi=180)
    plt.semilogy(df["layers"], df["mean_orthogonality_error"], marker="o", linewidth=2.5, label="mean")
    plt.semilogy(df["layers"], df["p90_orthogonality_error"], marker="s", linewidth=2.0, label="90th percentile")
    plt.xlabel("Deep CORO layers")
    plt.ylabel(r"$\|R^\top R-I\|_F$")
    plt.title("Deep CORO convergence versus layer count")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "coro_convergence_vs_layers.png")
    plt.savefig(out_dir / "coro_convergence_vs_layers.svg")
    plt.close()

    plt.figure(figsize=(7.5, 4.8), dpi=180)
    plt.plot(df["layers"], df["mean_angle_to_gt_deg"], marker="o", linewidth=2.5, label="to ground truth")
    plt.plot(df["layers"], df["mean_angle_to_svd_deg"], marker="s", linewidth=2.0, label="to SVD projection")
    plt.xlabel("Deep CORO layers")
    plt.ylabel("Mean rotation error (deg)")
    plt.title("Rotation accuracy after Deep CORO projection")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "coro_angle_vs_layers.png")
    plt.savefig(out_dir / "coro_angle_vs_layers.svg")
    plt.close()
    return df


def time_function(fn, repeats: int = 8, warmup: int = 3):
    for _ in range(warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


def runtime_experiment(out_dir: Path) -> pd.DataFrame:
    dtype = torch.float32
    batch_sizes = [32, 128, 512, 2048]
    L = 8
    rows = []
    for B in batch_sizes:
        M, _ = make_distorted_rotations(B, dtype=dtype, device=torch.device("cpu"), seed=7 + B)
        alpha = torch.ones(L, dtype=dtype)
        beta = torch.ones(L, dtype=dtype)
        # Pure PyTorch CORO CPU
        t_coro = time_function(lambda: deep_coro_forward_torch(M, alpha, beta), repeats=2, warmup=1)
        t_svd = time_function(lambda: svd_project_so3(M), repeats=1, warmup=1)
        rows.append(
            {
                "device": "cpu",
                "operator": "torch_deep_coro_reference",
                "batch": B,
                "layers": L,
                "median_time_ms": float(np.median(t_coro) * 1e3),
                "mean_time_ms": float(np.mean(t_coro) * 1e3),
                "median_us_per_matrix": float(np.median(t_coro) * 1e6 / B),
            }
        )
        rows.append(
            {
                "device": "cpu",
                "operator": "torch_svd_projection",
                "batch": B,
                "layers": 0,
                "median_time_ms": float(np.median(t_svd) * 1e3),
                "mean_time_ms": float(np.mean(t_svd) * 1e3),
                "median_us_per_matrix": float(np.median(t_svd) * 1e6 / B),
            }
        )

    # If CUDA is available and the extension was built, benchmark it too.
    if torch.cuda.is_available():
        for B in batch_sizes:
            M, _ = make_distorted_rotations(B, dtype=dtype, device=torch.device("cuda"), seed=17 + B)
            alpha = torch.ones(L, dtype=dtype, device="cuda")
            beta = torch.ones(L, dtype=dtype, device="cuda")
            try:
                with torch.no_grad():
                    t_cuda = time_function(lambda: deep_coro_forward(M, alpha, beta, use_extension=True), repeats=20, warmup=10)
                rows.append(
                    {
                        "device": "cuda",
                        "operator": "cuda_deep_coro_extension",
                        "batch": B,
                        "layers": L,
                        "median_time_ms": float(np.median(t_cuda) * 1e3),
                        "mean_time_ms": float(np.mean(t_cuda) * 1e3),
                        "median_us_per_matrix": float(np.median(t_cuda) * 1e6 / B),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "device": "cuda",
                        "operator": "cuda_deep_coro_extension_failed",
                        "batch": B,
                        "layers": L,
                        "median_time_ms": np.nan,
                        "mean_time_ms": np.nan,
                        "median_us_per_matrix": np.nan,
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operator_runtime.csv", index=False)

    plt.figure(figsize=(8.0, 5.0), dpi=180)
    for op, group in df[df["device"] == "cpu"].groupby("operator"):
        plt.loglog(group["batch"], group["median_time_ms"], marker="o", linewidth=2.4, label=op.replace("torch_", ""))
    cuda_group = df[df["device"] == "cuda"]
    if not cuda_group.empty and cuda_group["median_time_ms"].notna().any():
        for op, group in cuda_group.groupby("operator"):
            if group["median_time_ms"].notna().any():
                plt.loglog(group["batch"], group["median_time_ms"], marker="s", linewidth=2.4, label=op)
    plt.xlabel("Batch size")
    plt.ylabel("Median runtime (ms)")
    plt.title("Deep CORO operator runtime")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "operator_runtime.png")
    plt.savefig(out_dir / "operator_runtime.svg")
    plt.close()
    return df


def extension_consistency_experiment(out_dir: Path) -> pd.DataFrame:
    # CPU pure-torch versus extension if installed.  CUDA checked only when available.
    rows = []
    dtype = torch.float32
    B = 512
    L = 8
    M, _ = make_distorted_rotations(B, dtype=dtype, device=torch.device("cpu"), seed=321)
    alpha = torch.ones(L, dtype=dtype)
    beta = torch.ones(L, dtype=dtype)
    with torch.no_grad():
        R_torch = deep_coro_forward_torch(M, alpha, beta)
        try:
            R_ext_cpu = deep_coro_forward(M, alpha, beta, use_extension=True)
            err = (R_torch - R_ext_cpu).abs().max().item()
            rows.append({"path": "cpu_extension_vs_torch", "max_abs_difference": err, "available": 1})
        except Exception as exc:
            rows.append({"path": "cpu_extension_vs_torch", "max_abs_difference": np.nan, "available": 0, "error": str(exc)})

        if torch.cuda.is_available():
            try:
                M_cuda = M.cuda()
                alpha_cuda = alpha.cuda()
                beta_cuda = beta.cuda()
                R_cuda = deep_coro_forward(M_cuda, alpha_cuda, beta_cuda, use_extension=True).cpu()
                err_cuda = (R_torch - R_cuda).abs().max().item()
                rows.append({"path": "cuda_extension_vs_cpu_torch", "max_abs_difference": err_cuda, "available": 1})
            except Exception as exc:
                rows.append({"path": "cuda_extension_vs_cpu_torch", "max_abs_difference": np.nan, "available": 0, "error": str(exc)})
        else:
            rows.append({"path": "cuda_extension_vs_cpu_torch", "max_abs_difference": np.nan, "available": 0, "error": "CUDA not available in this environment"})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operator_consistency.csv", index=False)
    return df


def write_markdown_summary(out_dir: Path, conv: pd.DataFrame, runtime: pd.DataFrame, consistency: pd.DataFrame):
    lines = []
    lines.append("# Deep CORO CUDA operator experiment summary\n")
    lines.append("## Convergence versus layer count\n")
    lines.append(conv.to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n\n## Runtime table\n")
    lines.append(runtime.to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n\n## Extension consistency\n")
    lines.append(consistency.to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n\nNote: shipped results were generated in the current environment. If CUDA is not available, GPU timing rows are intentionally absent/unavailable.\n")
    (out_dir / "summary_tables.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        meta["device_name_0"] = torch.cuda.get_device_name(0)
    (out_dir / "environment.json").write_text(json.dumps(meta, indent=2))

    conv = convergence_experiment(out_dir)
    runtime = runtime_experiment(out_dir)
    consistency = extension_consistency_experiment(out_dir)
    write_markdown_summary(out_dir, conv, runtime, consistency)
    print("Wrote results to", out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
