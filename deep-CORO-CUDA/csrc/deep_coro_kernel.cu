#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {

template <typename scalar_t>
__device__ __forceinline__ void cofactor3_dev(const scalar_t x[9], scalar_t c[9]) {
    c[0] = x[4] * x[8] - x[5] * x[7];
    c[1] = x[5] * x[6] - x[3] * x[8];
    c[2] = x[3] * x[7] - x[4] * x[6];

    c[3] = x[2] * x[7] - x[1] * x[8];
    c[4] = x[0] * x[8] - x[2] * x[6];
    c[5] = x[1] * x[6] - x[0] * x[7];

    c[6] = x[1] * x[5] - x[2] * x[4];
    c[7] = x[2] * x[3] - x[0] * x[5];
    c[8] = x[0] * x[4] - x[1] * x[3];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t det3_dev(const scalar_t x[9]) {
    return x[0] * (x[4] * x[8] - x[5] * x[7])
         - x[1] * (x[3] * x[8] - x[5] * x[6])
         + x[2] * (x[3] * x[7] - x[4] * x[6]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t inv_cuberoot_abs(scalar_t v);

template <>
__device__ __forceinline__ float inv_cuberoot_abs<float>(float v) {
    return powf(v, -0.3333333333333333f);
}

template <>
__device__ __forceinline__ double inv_cuberoot_abs<double>(double v) {
    return pow(v, -0.3333333333333333333333333333);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t signed_inv_cuberoot_dev(scalar_t det, scalar_t eps) {
    scalar_t absdet = det >= scalar_t(0) ? det : -det;
    if (absdet < eps) absdet = eps;
    scalar_t scale = inv_cuberoot_abs<scalar_t>(absdet);
    return det >= scalar_t(0) ? scale : -scale;
}

template <typename scalar_t>
__global__ void deep_coro_forward_kernel(
    const scalar_t* __restrict__ M,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ out,
    int64_t B,
    int64_t L,
    scalar_t eps) {

    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= B) return;

    scalar_t x[9];
    scalar_t c[9];
    scalar_t y[9];

    const scalar_t* src = M + 9 * n;
    #pragma unroll
    for (int i = 0; i < 9; ++i) x[i] = src[i];

    for (int64_t ell = 0; ell < L; ++ell) {
        cofactor3_dev<scalar_t>(x, c);
        const scalar_t aa = alpha[ell];
        const scalar_t bb = beta[ell];
        #pragma unroll
        for (int i = 0; i < 9; ++i) y[i] = aa * x[i] + bb * c[i];
        scalar_t det = det3_dev<scalar_t>(y);
        scalar_t rho = signed_inv_cuberoot_dev<scalar_t>(det, eps);
        #pragma unroll
        for (int i = 0; i < 9; ++i) x[i] = rho * y[i];
    }

    scalar_t* dst = out + 9 * n;
    #pragma unroll
    for (int i = 0; i < 9; ++i) dst[i] = x[i];
}

} // namespace

torch::Tensor deep_coro_forward_cuda(torch::Tensor M, torch::Tensor alpha, torch::Tensor beta, double eps) {
    TORCH_CHECK(M.is_cuda(), "M must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda() && beta.is_cuda(), "alpha and beta must be CUDA tensors");
    TORCH_CHECK(M.dim() == 3 && M.size(1) == 3 && M.size(2) == 3, "M must be [B,3,3]");
    TORCH_CHECK(alpha.dim() == 1 && beta.dim() == 1, "alpha and beta must be 1-D tensors");
    TORCH_CHECK(alpha.numel() == beta.numel(), "alpha and beta must have the same length");
    TORCH_CHECK(M.scalar_type() == alpha.scalar_type() && M.scalar_type() == beta.scalar_type(),
                "M, alpha and beta must have the same dtype");

    auto Mc = M.contiguous();
    auto ac = alpha.contiguous();
    auto bc = beta.contiguous();
    auto out = torch::empty_like(Mc);

    const int64_t B = Mc.size(0);
    const int64_t L = ac.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((B + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(Mc.scalar_type(), "deep_coro_forward_cuda", [&](){
        deep_coro_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            Mc.data_ptr<scalar_t>(),
            ac.data_ptr<scalar_t>(),
            bc.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            L,
            static_cast<scalar_t>(eps));
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
