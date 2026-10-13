#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Evaluates a uniform cubic spline on GPU matching SciPy CubicSpline
// coeffs has layout [4, n_intervals], with coeffs[0]=c0 (t^3), coeffs[1]=c1 (t^2), coeffs[2]=c2 (t^1), coeffs[3]=c3 (constant)
__device__ __forceinline__ double eval_uniform_cubic_spline(
    double r,
    double dr,
    double rcut,
    int n_intervals,
    const double* __restrict__ coeffs_4_x_n  // leading dimension 4: stride across degrees is n_intervals
) {
    if (r < 0.0 || r > rcut) {
        return 0.0;
    }
    int k = (int)floor(r / dr);
    if (k < 0) k = 0;
    if (k >= n_intervals) k = n_intervals - 1;
    double t = r - (double)k * dr;

    // Horner evaluation: c3 + t * (c2 + t * (c1 + t * c0))
    double c0 = coeffs_4_x_n[0 * n_intervals + k];
    double c1 = coeffs_4_x_n[1 * n_intervals + k];
    double c2 = coeffs_4_x_n[2 * n_intervals + k];
    double c3 = coeffs_4_x_n[3 * n_intervals + k];

    return c3 + t * (c2 + t * (c1 + t * c0));
}
