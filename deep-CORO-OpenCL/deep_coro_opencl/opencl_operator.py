from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class OpenCLUnavailableError(RuntimeError):
    """Raised when PyOpenCL or a valid OpenCL platform/device is unavailable."""


_KERNEL_PATH = Path(__file__).resolve().parent / "kernels" / "deep_coro_so3.cl"


def _load_pyopencl():
    try:
        import pyopencl as cl  # type: ignore
        return cl
    except Exception as exc:  # pragma: no cover - depends on host
        raise OpenCLUnavailableError(
            "PyOpenCL is not installed. Install with `pip install pyopencl`, "
            "or use the NumPy/PyTorch reference path."
        ) from exc


class DeepCoroOpenCLOperator:
    """Batched Deep CORO inference operator implemented in OpenCL.

    The operator is intentionally forward-only. Training should use a differentiable
    Python/PyTorch reference implementation, while this class is intended for fast
    inference and deployment-time benchmarking across OpenCL devices.
    """

    def __init__(
        self,
        platform_index: int = 0,
        device_index: int = 0,
        fast_math: bool = False,
        profiling: bool = True,
        build_options: Optional[List[str]] = None,
    ) -> None:
        self.cl = _load_pyopencl()
        platforms = self.cl.get_platforms()
        if not platforms:
            raise OpenCLUnavailableError("No OpenCL platforms were found on this machine.")
        if platform_index >= len(platforms):
            raise OpenCLUnavailableError(f"Requested platform {platform_index}, but only {len(platforms)} exist.")
        devices = platforms[platform_index].get_devices()
        if not devices:
            raise OpenCLUnavailableError(f"Platform {platform_index} has no OpenCL devices.")
        if device_index >= len(devices):
            raise OpenCLUnavailableError(f"Requested device {device_index}, but only {len(devices)} exist.")
        self.platform = platforms[platform_index]
        self.device = devices[device_index]
        self.context = self.cl.Context([self.device])
        props = 0
        if profiling:
            props |= self.cl.command_queue_properties.PROFILING_ENABLE
        self.queue = self.cl.CommandQueue(self.context, properties=props)
        source = _KERNEL_PATH.read_text(encoding="utf-8")
        opts = ["-w"]
        if fast_math:
            opts += ["-cl-mad-enable", "-cl-fast-relaxed-math"]
        if build_options:
            opts += list(build_options)
        self.program = self.cl.Program(self.context, source).build(options=opts)
        self.fast_math = fast_math
        self.profiling = profiling
        self.build_options = opts

    def info(self) -> dict:
        return {
            "platform": self.platform.name,
            "device": self.device.name,
            "device_type": str(self.device.type),
            "max_compute_units": getattr(self.device, "max_compute_units", None),
            "global_mem_size": getattr(self.device, "global_mem_size", None),
            "build_options": " ".join(self.build_options),
        }

    def forward(
        self,
        M: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-12,
        return_event_time: bool = False,
    ):
        cl = self.cl
        M = np.asarray(M)
        if M.ndim != 3 or M.shape[1:] != (3, 3):
            raise ValueError("M must have shape (B,3,3)")
        if M.dtype not in (np.float32, np.float64):
            M = M.astype(np.float32)
        dtype = M.dtype
        alpha = np.asarray(alpha, dtype=dtype)
        beta = np.asarray(beta, dtype=dtype)
        if alpha.ndim != 1 or beta.ndim != 1 or alpha.size != beta.size:
            raise ValueError("alpha and beta must be 1-D arrays with equal length")
        B = np.int32(M.shape[0])
        L = np.int32(alpha.size)
        mf = cl.mem_flags
        M_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(M))
        a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(alpha))
        b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(beta))
        out = np.empty_like(M)
        out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, out.nbytes)
        kernel_name = "deep_coro_so3_f32" if dtype == np.float32 else "deep_coro_so3_f64"
        kernel = getattr(self.program, kernel_name)
        global_size = (int(B),)
        local_size = None
        eps_arg = np.float32(eps) if dtype == np.float32 else np.float64(eps)
        event = kernel(self.queue, global_size, local_size, M_buf, a_buf, b_buf, out_buf, L, eps_arg)
        cl.enqueue_copy(self.queue, out, out_buf).wait()
        elapsed_ms = None
        if return_event_time:
            try:
                event.wait()
                elapsed_ms = 1e-6 * (event.profile.end - event.profile.start)
            except Exception:
                elapsed_ms = None
        if return_event_time:
            return out, elapsed_ms
        return out


def list_opencl_devices() -> List[dict]:
    cl = _load_pyopencl()
    rows = []
    for pi, p in enumerate(cl.get_platforms()):
        for di, d in enumerate(p.get_devices()):
            rows.append(
                {
                    "platform_index": pi,
                    "device_index": di,
                    "platform": p.name,
                    "device": d.name,
                    "max_compute_units": getattr(d, "max_compute_units", None),
                    "global_mem_size": getattr(d, "global_mem_size", None),
                }
            )
    return rows
