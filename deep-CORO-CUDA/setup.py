import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import torch


def get_extensions():
    sources = ["csrc/deep_coro.cpp"]
    define_macros = []
    extra_compile_args = {"cxx": ["-O3", "-w"]}

    force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
    use_cuda = (torch.cuda.is_available() or force_cuda) and CUDA_HOME is not None

    if use_cuda:
        sources.append("csrc/deep_coro_kernel.cu")
        define_macros.append(("WITH_CUDA", None))
        nvcc_flags = ["-O3", "-w"]
        if os.environ.get("CORO_NO_FAST_MATH", "0") != "1":
            nvcc_flags.append("--use_fast_math")
        extra_compile_args["nvcc"] = nvcc_flags
        ext_cls = CUDAExtension
    else:
        ext_cls = CppExtension

    return [
        ext_cls(
            name="deep_coro_ext",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="deep_coro_cuda",
    version="0.1.0",
    description="Batched CUDA/C++ operator for Deep CORO on SO(3)",
    packages=find_packages(),
    python_requires=">=3.9",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
