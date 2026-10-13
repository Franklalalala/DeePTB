#pragma once
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ double compute_real_ylm_abacus(int l, int m, double x, double y, double z) {
    double r2 = x * x + y * y + z * z;
    double r = sqrt(r2);
    if (r < 1e-14) {
        return (l == 0) ? (0.5 / sqrt(M_PI)) : 0.0;
    }
    double costheta = z / r;
    if (costheta > 1.0) costheta = 1.0;
    if (costheta < -1.0) costheta = -1.0;
    double sintheta = sqrt(fmax(0.0, 1.0 - costheta * costheta));
    double phi = atan2(y, x);

    int am = (m >= 0) ? m : -m;
    double pmm = 1.0;
    if (am > 0) {
        double somx2 = sintheta;
        double fact = 1.0;
        for (int i = 1; i <= am; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    double plm;
    if (l == am) {
        plm = pmm;
    } else if (l == am + 1) {
        plm = costheta * (2 * am + 1) * pmm;
    } else {
        double p_prev = pmm;
        double p_curr = costheta * (2 * am + 1) * pmm;
        for (int ll = am + 2; ll <= l; ++ll) {
            double p_next = (costheta * (2 * ll - 1) * p_curr - (ll + am - 1) * p_prev) / (ll - am);
            p_prev = p_curr;
            p_curr = p_next;
        }
        plm = p_curr;
    }

    // Normalization factor: sqrt((2l+1)/(4pi) * (l-am)! / (l+am)!)
    double factorial_ratio = 1.0;
    for (int k = l - am + 1; k <= l + am; ++k) {
        factorial_ratio /= (double)k;
    }
    double norm = sqrt((2.0 * l + 1.0) / (4.0 * M_PI) * factorial_ratio);

    if (m == 0) {
        return norm * plm;
    } else if (m > 0) {
        return sqrt(2.0) * norm * plm * cos((double)am * phi);
    } else {
        return sqrt(2.0) * norm * plm * sin((double)am * phi);
    }
}
