"""OpenCL and reference operators for Deep CORO on SO(3)."""
from .reference import (
    cofactor3_np,
    deep_coro_forward_np,
    random_rotations,
    svd_project_so3,
    rotation_error_deg,
    orthogonality_error,
)
from .opencl_operator import DeepCoroOpenCLOperator, OpenCLUnavailableError

__all__ = [
    "cofactor3_np",
    "deep_coro_forward_np",
    "random_rotations",
    "svd_project_so3",
    "rotation_error_deg",
    "orthogonality_error",
    "DeepCoroOpenCLOperator",
    "OpenCLUnavailableError",
]
