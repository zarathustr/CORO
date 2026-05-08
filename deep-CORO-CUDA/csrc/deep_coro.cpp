#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

#ifdef WITH_CUDA
torch::Tensor deep_coro_forward_cuda(torch::Tensor M, torch::Tensor alpha, torch::Tensor beta, double eps);
#endif

namespace {

template <typename scalar_t>
inline void cofactor3_local(const scalar_t x[9], scalar_t c[9]) {
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
inline scalar_t det3_local(const scalar_t x[9]) {
    return x[0] * (x[4] * x[8] - x[5] * x[7])
         - x[1] * (x[3] * x[8] - x[5] * x[6])
         + x[2] * (x[3] * x[7] - x[4] * x[6]);
}

template <typename scalar_t>
inline scalar_t signed_inv_cuberoot(scalar_t det, scalar_t eps) {
    scalar_t absdet = det >= scalar_t(0) ? det : -det;
    if (absdet < eps) absdet = eps;
    scalar_t scale = std::pow(absdet, scalar_t(-1.0 / 3.0));
    return det >= scalar_t(0) ? scale : -scale;
}

torch::Tensor deep_coro_forward_cpu(torch::Tensor M, torch::Tensor alpha, torch::Tensor beta, double eps) {
    TORCH_CHECK(M.device().is_cpu(), "CPU implementation received non-CPU tensor");
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

    AT_DISPATCH_FLOATING_TYPES(Mc.scalar_type(), "deep_coro_forward_cpu", [&](){
        const scalar_t* m_ptr = Mc.data_ptr<scalar_t>();
        const scalar_t* a_ptr = ac.data_ptr<scalar_t>();
        const scalar_t* b_ptr = bc.data_ptr<scalar_t>();
        scalar_t* o_ptr = out.data_ptr<scalar_t>();
        const scalar_t eps_t = static_cast<scalar_t>(eps);

        at::parallel_for(0, B, 1024, [&](int64_t begin, int64_t end) {
            scalar_t x[9];
            scalar_t c[9];
            scalar_t y[9];
            for (int64_t n = begin; n < end; ++n) {
                const scalar_t* src = m_ptr + 9 * n;
                for (int i = 0; i < 9; ++i) x[i] = src[i];

                for (int64_t ell = 0; ell < L; ++ell) {
                    cofactor3_local<scalar_t>(x, c);
                    const scalar_t aa = a_ptr[ell];
                    const scalar_t bb = b_ptr[ell];
                    for (int i = 0; i < 9; ++i) y[i] = aa * x[i] + bb * c[i];
                    scalar_t det = det3_local<scalar_t>(y);
                    scalar_t rho = signed_inv_cuberoot<scalar_t>(det, eps_t);
                    for (int i = 0; i < 9; ++i) x[i] = rho * y[i];
                }

                scalar_t* dst = o_ptr + 9 * n;
                for (int i = 0; i < 9; ++i) dst[i] = x[i];
            }
        });
    });

    return out;
}

} // namespace

torch::Tensor deep_coro_forward(torch::Tensor M, torch::Tensor alpha, torch::Tensor beta, double eps) {
    if (M.is_cuda()) {
#ifdef WITH_CUDA
        return deep_coro_forward_cuda(M, alpha, beta, eps);
#else
        TORCH_CHECK(false, "deep_coro_ext was built without CUDA support");
#endif
    }
    return deep_coro_forward_cpu(M, alpha, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deep_coro_forward", &deep_coro_forward,
          "Batched Deep CORO forward projection on SO(3) (CPU/CUDA)",
          py::arg("M"), py::arg("alpha"), py::arg("beta"), py::arg("eps") = 1e-12);
}
