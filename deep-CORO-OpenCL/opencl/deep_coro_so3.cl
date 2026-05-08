// Deep CORO SO(3) OpenCL kernels.
// Each work-item processes one 3x3 matrix and keeps all CORO state in private registers.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline void cofactor3_f32(const float x[9], float c[9]) {
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

inline float det3_f32(const float x[9]) {
    return x[0] * (x[4] * x[8] - x[5] * x[7])
         - x[1] * (x[3] * x[8] - x[5] * x[6])
         + x[2] * (x[3] * x[7] - x[4] * x[6]);
}

__kernel void deep_coro_so3_f32(
    __global const float * restrict M,
    __global const float * restrict alpha,
    __global const float * restrict beta,
    __global float * restrict out,
    const int L,
    const float eps) {

    const int gid = get_global_id(0);
    float x[9];
    float c[9];
    float y[9];

    const int base = 9 * gid;
    #pragma unroll
    for (int i = 0; i < 9; ++i) x[i] = M[base + i];

    for (int ell = 0; ell < L; ++ell) {
        cofactor3_f32(x, c);
        const float aa = alpha[ell];
        const float bb = beta[ell];
        #pragma unroll
        for (int i = 0; i < 9; ++i) y[i] = aa * x[i] + bb * c[i];
        float det = det3_f32(y);
        float absdet = fabs(det);
        absdet = fmax(absdet, eps);
        float rho = pow(absdet, -0.3333333333333333f);
        if (det < 0.0f) rho = -rho;
        #pragma unroll
        for (int i = 0; i < 9; ++i) x[i] = rho * y[i];
    }

    #pragma unroll
    for (int i = 0; i < 9; ++i) out[base + i] = x[i];
}

inline void cofactor3_f64(const double x[9], double c[9]) {
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

inline double det3_f64(const double x[9]) {
    return x[0] * (x[4] * x[8] - x[5] * x[7])
         - x[1] * (x[3] * x[8] - x[5] * x[6])
         + x[2] * (x[3] * x[7] - x[4] * x[6]);
}

__kernel void deep_coro_so3_f64(
    __global const double * restrict M,
    __global const double * restrict alpha,
    __global const double * restrict beta,
    __global double * restrict out,
    const int L,
    const double eps) {

    const int gid = get_global_id(0);
    double x[9];
    double c[9];
    double y[9];

    const int base = 9 * gid;
    #pragma unroll
    for (int i = 0; i < 9; ++i) x[i] = M[base + i];

    for (int ell = 0; ell < L; ++ell) {
        cofactor3_f64(x, c);
        const double aa = alpha[ell];
        const double bb = beta[ell];
        #pragma unroll
        for (int i = 0; i < 9; ++i) y[i] = aa * x[i] + bb * c[i];
        double det = det3_f64(y);
        double absdet = fabs(det);
        absdet = fmax(absdet, eps);
        double rho = pow(absdet, -0.3333333333333333333333333333333);
        if (det < 0.0) rho = -rho;
        #pragma unroll
        for (int i = 0; i < 9; ++i) x[i] = rho * y[i];
    }

    #pragma unroll
    for (int i = 0; i < 9; ++i) out[base + i] = x[i];
}
