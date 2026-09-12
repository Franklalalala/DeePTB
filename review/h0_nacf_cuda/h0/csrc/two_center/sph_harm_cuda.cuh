#ifndef SPH_HARM_CUDA_CUH_
#define SPH_HARM_CUDA_CUH_

#include <cuda_runtime.h>
#include <cmath>

namespace two_center {

// Constant array of ylmcoef from ABACUS ModuleBase::Ylm
__constant__ const double c_ylmcoef[36] = {
    0.28209479177387814, // 1/sqrt(4*pi)
    0.48860251190291992, // sqrt(3/(4*pi))
    1.9364916731037085, // sqrt(15)/2
    1.1180339887498949, // sqrt(5)/2
    2.2360679774997898, // sqrt(5)
    0.57735026918962584, // 1/sqrt(3)
    1.2909944487358056, // sqrt(5/3)
    1.9720265943665387, // sqrt(35/9)
    1.0183501544346312, // sqrt(7/3)/1.5
    2.0916500663351889, // sqrt(35/8)
    0.93541434669348533, // sqrt(7/8)
    2.6457513110645907, // sqrt(7)
    0.2581988897471611, // 1/sqrt(15)
    0.96609178307929588, // sqrt(14/15)
    2.1602468994692869, // sqrt(14/3)
    1.984313483298443, // sqrt(7)*3/4
    1.0062305898749053, // 9/4/sqrt(5)
    2.0493901531919199, // sqrt(21/5)
    0.9797958971132712, // sqrt(24/25)
    2.2912878474779199, // sqrt(21)/2
    0.8660254037844386, // sqrt(3)/2
    0.1889822365046136, // 0.5/sqrt(7)
    0.98198050606196563, // 1.5*sqrt(3/7)
    2.1213203435596424, // 3/sqrt(2)
    1.9899748742132397, // 0.6*sqrt(11)
    1.002853072844814, // 0.8*sqrt(11/7)
    2.0310096011589902, // sqrt(33/8)
    0.99103120896511487, // sqrt(55/56)
    2.1712405933672376, // sqrt(33/7)
    0.94760708295868568, // sqrt(11)*2/7
    2.4874685927665499, // sqrt(11)*0.75
    0.82915619758884995, // sqrt(11)*0.25
    3.3166247903553998, // sqrt(11)
    0.14907119849998596, // 1/3/sqrt(5)
    0.98882646494608839, // 2/3*sqrt(11/5)
    2.0976176963403033, // sqrt(22/5)
};

// Host copy of ylmcoef
inline const double* get_host_ylmcoef() {
    static const double h_ylmcoef[36] = {
    0.28209479177387814, // 1/sqrt(4*pi)
    0.48860251190291992, // sqrt(3/(4*pi))
    1.9364916731037085, // sqrt(15)/2
    1.1180339887498949, // sqrt(5)/2
    2.2360679774997898, // sqrt(5)
    0.57735026918962584, // 1/sqrt(3)
    1.2909944487358056, // sqrt(5/3)
    1.9720265943665387, // sqrt(35/9)
    1.0183501544346312, // sqrt(7/3)/1.5
    2.0916500663351889, // sqrt(35/8)
    0.93541434669348533, // sqrt(7/8)
    2.6457513110645907, // sqrt(7)
    0.2581988897471611, // 1/sqrt(15)
    0.96609178307929588, // sqrt(14/15)
    2.1602468994692869, // sqrt(14/3)
    1.984313483298443, // sqrt(7)*3/4
    1.0062305898749053, // 9/4/sqrt(5)
    2.0493901531919199, // sqrt(21/5)
    0.9797958971132712, // sqrt(24/25)
    2.2912878474779199, // sqrt(21)/2
    0.8660254037844386, // sqrt(3)/2
    0.1889822365046136, // 0.5/sqrt(7)
    0.98198050606196563, // 1.5*sqrt(3/7)
    2.1213203435596424, // 3/sqrt(2)
    1.9899748742132397, // 0.6*sqrt(11)
    1.002853072844814, // 0.8*sqrt(11/7)
    2.0310096011589902, // sqrt(33/8)
    0.99103120896511487, // sqrt(55/56)
    2.1712405933672376, // sqrt(33/7)
    0.94760708295868568, // sqrt(11)*2/7
    2.4874685927665499, // sqrt(11)*0.75
    0.82915619758884995, // sqrt(11)*0.25
    3.3166247903553998, // sqrt(11)
    0.14907119849998596, // 1/3/sqrt(5)
    0.98882646494608839, // 2/3*sqrt(11/5)
    2.0976176963403033, // sqrt(22/5)
};
    return h_ylmcoef;
}

__host__ __device__ inline int ylm_index(int l, int m) {
    return l * l + (m > 0 ? 2 * m - 1 : -2 * m);
}

// Compute solid spherical harmonics R^l * Y_{lm}(vR) on GPU
__device__ inline void compute_rl_sph_harm_device(
    int Lmax,
    double x,
    double y,
    double z,
    double* rly
) {
    double radius2 = x * x + y * y + z * z;

    // L = 0
    rly[0] = c_ylmcoef[0];
    if (Lmax == 0) return;

    // L = 1
    rly[1] = c_ylmcoef[1] * z;
    rly[2] = -c_ylmcoef[1] * x;
    rly[3] = -c_ylmcoef[1] * y;
    if (Lmax == 1) return;

    // L = 2
    rly[4] = c_ylmcoef[2] * z * rly[1] - c_ylmcoef[3] * rly[0] * radius2;
    double tmp0 = c_ylmcoef[4] * z;
    rly[5] = tmp0 * rly[2];
    rly[6] = tmp0 * rly[3];
    double tmp2 = c_ylmcoef[4] * x;
    rly[7] = c_ylmcoef[5] * rly[4] - c_ylmcoef[6] * rly[0] * radius2 - tmp2 * rly[2];
    rly[8] = -tmp2 * rly[3];
    if (Lmax == 2) return;

    // L = 3
    rly[9] = c_ylmcoef[7] * z * rly[4] - c_ylmcoef[8] * rly[1] * radius2;
    double tmp3 = c_ylmcoef[9] * z;
    rly[10] = tmp3 * rly[5] - c_ylmcoef[10] * rly[2] * radius2;
    rly[11] = tmp3 * rly[6] - c_ylmcoef[10] * rly[3] * radius2;
    double tmp4 = c_ylmcoef[11] * z;
    rly[12] = tmp4 * rly[7];
    rly[13] = tmp4 * rly[8];
    double tmp5 = c_ylmcoef[14] * x;
    rly[14] = c_ylmcoef[12] * rly[10] - c_ylmcoef[13] * rly[2] * radius2 - tmp5 * rly[7];
    rly[15] = c_ylmcoef[12] * rly[11] - c_ylmcoef[13] * rly[3] * radius2 - tmp5 * rly[8];
    if (Lmax == 3) return;

    // L = 4
    rly[16] = c_ylmcoef[15] * z * rly[9] - c_ylmcoef[16] * rly[4] * radius2;
    double tmp6 = c_ylmcoef[17] * z;
    rly[17] = tmp6 * rly[10] - c_ylmcoef[18] * rly[5] * radius2;
    rly[18] = tmp6 * rly[11] - c_ylmcoef[18] * rly[6] * radius2;
    double tmp7 = c_ylmcoef[19] * z;
    rly[19] = tmp7 * rly[12] - c_ylmcoef[20] * rly[7] * radius2;
    rly[20] = tmp7 * rly[13] - c_ylmcoef[20] * rly[8] * radius2;
    double tmp8 = 3.0 * z;
    rly[21] = tmp8 * rly[14];
    rly[22] = tmp8 * rly[15];
    double tmp9 = c_ylmcoef[23] * x;
    rly[23] = c_ylmcoef[21] * rly[19] - c_ylmcoef[22] * rly[7] * radius2 - tmp9 * rly[14];
    rly[24] = c_ylmcoef[21] * rly[20] - c_ylmcoef[22] * rly[8] * radius2 - tmp9 * rly[15];
    if (Lmax == 4) return;

    // L = 5
    rly[25] = c_ylmcoef[24] * z * rly[16] - c_ylmcoef[25] * rly[9] * radius2;
    double tmp10 = c_ylmcoef[26] * z;
    rly[26] = tmp10 * rly[17] - c_ylmcoef[27] * rly[10] * radius2;
    rly[27] = tmp10 * rly[18] - c_ylmcoef[27] * rly[11] * radius2;
    double tmp11 = c_ylmcoef[28] * z;
    rly[28] = tmp11 * rly[19] - c_ylmcoef[29] * rly[12] * radius2;
    rly[29] = tmp11 * rly[20] - c_ylmcoef[29] * rly[13] * radius2;
    double tmp12 = c_ylmcoef[30] * z;
    rly[30] = tmp12 * rly[21] - c_ylmcoef[31] * rly[14] * radius2;
    rly[31] = tmp12 * rly[22] - c_ylmcoef[31] * rly[15] * radius2;
    double tmp13 = c_ylmcoef[32] * z;
    rly[32] = tmp13 * rly[23];
    rly[33] = tmp13 * rly[24];
    double tmp14 = c_ylmcoef[35] * x;
    rly[34] = c_ylmcoef[33] * rly[30] - c_ylmcoef[34] * rly[14] * radius2 - tmp14 * rly[23];
    rly[35] = c_ylmcoef[33] * rly[31] - c_ylmcoef[34] * rly[15] * radius2 - tmp14 * rly[24];
    if (Lmax == 5) return;

    // L >= 6
    for (int il = 6; il <= Lmax; ++il) {
        int istart = il * il;
        int istart1 = (il - 1) * (il - 1);
        int istart2 = (il - 2) * (il - 2);

        double fac2 = sqrt(4.0 * istart - 1.0);
        double fac4 = sqrt(4.0 * istart1 - 1.0);

        for (int im = 0; im < 2 * il - 1; ++im) {
            int imm = (im + 1) / 2;
            rly[istart + im] = (fac2 / sqrt((double)istart - imm * imm)) *
                (z * rly[istart1 + im] - (sqrt((double)istart1 - imm * imm) / fac4) * rly[istart2 + im] * radius2);
        }

        double bl1 = sqrt(2.0 * il / (2.0 * il + 1.0));
        double bl2 = sqrt((2.0 * il - 2.0) / (2.0 * il - 1.0));
        double bl3 = sqrt(2.0) / fac2;

        rly[istart + 2 * il - 1] = (bl3 * rly[istart + 2 * il - 5] - bl2 * rly[istart2 + 2 * il - 5] * radius2 - 2.0 * x * rly[istart1 + 2 * il - 3]) / bl1;
        rly[istart + 2 * il] = (bl3 * rly[istart + 2 * il - 4] - bl2 * rly[istart2 + 2 * il - 4] * radius2 - 2.0 * x * rly[istart1 + 2 * il - 2]) / bl1;
    }
}

// Host version for CPU validation
inline void compute_rl_sph_harm_host(
    int Lmax,
    double x,
    double y,
    double z,
    double* rly
) {
    const double* h_coef = get_host_ylmcoef();
    double radius2 = x * x + y * y + z * z;

    rly[0] = h_coef[0];
    if (Lmax == 0) return;

    rly[1] = h_coef[1] * z;
    rly[2] = -h_coef[1] * x;
    rly[3] = -h_coef[1] * y;
    if (Lmax == 1) return;

    rly[4] = h_coef[2] * z * rly[1] - h_coef[3] * rly[0] * radius2;
    double tmp0 = h_coef[4] * z;
    rly[5] = tmp0 * rly[2];
    rly[6] = tmp0 * rly[3];
    double tmp2 = h_coef[4] * x;
    rly[7] = h_coef[5] * rly[4] - h_coef[6] * rly[0] * radius2 - tmp2 * rly[2];
    rly[8] = -tmp2 * rly[3];
    if (Lmax == 2) return;

    rly[9] = h_coef[7] * z * rly[4] - h_coef[8] * rly[1] * radius2;
    double tmp3 = h_coef[9] * z;
    rly[10] = tmp3 * rly[5] - h_coef[10] * rly[2] * radius2;
    rly[11] = tmp3 * rly[6] - h_coef[10] * rly[3] * radius2;
    double tmp4 = h_coef[11] * z;
    rly[12] = tmp4 * rly[7];
    rly[13] = tmp4 * rly[8];
    double tmp5 = h_coef[14] * x;
    rly[14] = h_coef[12] * rly[10] - h_coef[13] * rly[2] * radius2 - tmp5 * rly[7];
    rly[15] = h_coef[12] * rly[11] - h_coef[13] * rly[3] * radius2 - tmp5 * rly[8];
    if (Lmax == 3) return;

    rly[16] = h_coef[15] * z * rly[9] - h_coef[16] * rly[4] * radius2;
    double tmp6 = h_coef[17] * z;
    rly[17] = tmp6 * rly[10] - h_coef[18] * rly[5] * radius2;
    rly[18] = tmp6 * rly[11] - h_coef[18] * rly[6] * radius2;
    double tmp7 = h_coef[19] * z;
    rly[19] = tmp7 * rly[12] - h_coef[20] * rly[7] * radius2;
    rly[20] = tmp7 * rly[13] - h_coef[20] * rly[8] * radius2;
    double tmp8 = 3.0 * z;
    rly[21] = tmp8 * rly[14];
    rly[22] = tmp8 * rly[15];
    double tmp9 = h_coef[23] * x;
    rly[23] = h_coef[21] * rly[19] - h_coef[22] * rly[7] * radius2 - tmp9 * rly[14];
    rly[24] = h_coef[21] * rly[20] - h_coef[22] * rly[8] * radius2 - tmp9 * rly[15];
    if (Lmax == 4) return;

    rly[25] = h_coef[24] * z * rly[16] - h_coef[25] * rly[9] * radius2;
    double tmp10 = h_coef[26] * z;
    rly[26] = tmp10 * rly[17] - h_coef[27] * rly[10] * radius2;
    rly[27] = tmp10 * rly[18] - h_coef[27] * rly[11] * radius2;
    double tmp11 = h_coef[28] * z;
    rly[28] = tmp11 * rly[19] - h_coef[29] * rly[12] * radius2;
    rly[29] = tmp11 * rly[20] - h_coef[29] * rly[13] * radius2;
    double tmp12 = h_coef[30] * z;
    rly[30] = tmp12 * rly[21] - h_coef[31] * rly[14] * radius2;
    rly[31] = tmp12 * rly[22] - h_coef[31] * rly[15] * radius2;
    double tmp13 = h_coef[32] * z;
    rly[32] = tmp13 * rly[23];
    rly[33] = tmp13 * rly[24];
    double tmp14 = h_coef[35] * x;
    rly[34] = h_coef[33] * rly[30] - h_coef[34] * rly[14] * radius2 - tmp14 * rly[23];
    rly[35] = h_coef[33] * rly[31] - h_coef[34] * rly[15] * radius2 - tmp14 * rly[24];
    if (Lmax == 5) return;

    for (int il = 6; il <= Lmax; ++il) {
        int istart = il * il;
        int istart1 = (il - 1) * (il - 1);
        int istart2 = (il - 2) * (il - 2);

        double fac2 = sqrt(4.0 * istart - 1.0);
        double fac4 = sqrt(4.0 * istart1 - 1.0);

        for (int im = 0; im < 2 * il - 1; ++im) {
            int imm = (im + 1) / 2;
            rly[istart + im] = (fac2 / sqrt((double)istart - imm * imm)) *
                (z * rly[istart1 + im] - (sqrt((double)istart1 - imm * imm) / fac4) * rly[istart2 + im] * radius2);
        }

        double bl1 = sqrt(2.0 * il / (2.0 * il + 1.0));
        double bl2 = sqrt((2.0 * il - 2.0) / (2.0 * il - 1.0));
        double bl3 = sqrt(2.0) / fac2;

        rly[istart + 2 * il - 1] = (bl3 * rly[istart + 2 * il - 5] - bl2 * rly[istart2 + 2 * il - 5] * radius2 - 2.0 * x * rly[istart1 + 2 * il - 3]) / bl1;
        rly[istart + 2 * il] = (bl3 * rly[istart + 2 * il - 4] - bl2 * rly[istart2 + 2 * il - 4] * radius2 - 2.0 * x * rly[istart1 + 2 * il - 2]) / bl1;
    }
}

} // namespace two_center

#endif // SPH_HARM_CUDA_CUH_