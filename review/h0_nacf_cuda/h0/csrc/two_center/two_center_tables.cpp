#include "two_center_tables.h"

namespace two_center {

py::dict extract_radial_table(py::object intor_obj) {
    auto* inst = reinterpret_cast<pybind11::detail::instance*>(intor_obj.ptr());
    void* vptr = inst->get_value_and_holder().value_ptr();
    if (!vptr) {
        throw std::runtime_error("Invalid or null TwoCenterIntegrator instance pointer");
    }
    auto* intor = reinterpret_cast<TwoCenterIntegrator*>(vptr);
    const auto& layout = get_table_layout(*intor);
    
    int ntab = layout.ntab_;
    int nr = layout.nr_;
    double rmax = layout.rmax_;
    double dr = (nr > 1) ? (rmax / (nr - 1)) : 0.0;
    
    auto tab = torch::from_blob(const_cast<double*>(layout.table_.data<double>()), {ntab, nr}, torch::kFloat64).clone();
    auto dtab = torch::from_blob(const_cast<double*>(layout.dtable_.data<double>()), {ntab, nr}, torch::kFloat64).clone();
    
    int ndim = layout.index_map_.shape().ndim();
    std::vector<int64_t> dims;
    for (int i = 0; i < ndim; ++i) {
        dims.push_back(layout.index_map_.shape().dim_size(i));
    }
    auto idx_map = torch::from_blob(const_cast<int*>(layout.index_map_.data<int>()), dims, torch::kInt32).clone();
    
    // Precompute cubic polynomial coefficients c0, c1, c2, c3 for every segment p in [0, nr-2]
    // Polynomial: S(R)/R^l = ((c3 * w + c2) * w + c1) * w + c0, where w = R - p * dr
    auto coeffs = torch::zeros({ntab, nr - 1, 4}, torch::kFloat64);
    double* c_ptr = coeffs.data_ptr<double>();
    const double* y = tab.data_ptr<double>();
    const double* dy = dtab.data_ptr<double>();
    double inv_dr = 1.0 / dr;
    double inv_dr2 = inv_dr * inv_dr;
    
    for (int itab = 0; itab < ntab; ++itab) {
        const double* y_row = y + itab * nr;
        const double* dy_row = dy + itab * nr;
        double* out_row = c_ptr + itab * (nr - 1) * 4;
        for (int p = 0; p < nr - 1; ++p) {
            double dd = (y_row[p + 1] - y_row[p]) * inv_dr;
            double c0 = y_row[p];
            double c1 = dy_row[p];
            double c3 = (c1 + dy_row[p + 1] - 2.0 * dd) * inv_dr2;
            double c2 = (dd - c1) * inv_dr - c3 * dr;
            out_row[p * 4 + 0] = c0;
            out_row[p * 4 + 1] = c1;
            out_row[p * 4 + 2] = c2;
            out_row[p * 4 + 3] = c3;
        }
    }
    
    py::dict res;
    res["table"] = tab;
    res["dtable"] = dtab;
    res["index_map"] = idx_map;
    res["coeffs"] = coeffs;
    res["nr"] = nr;
    res["ntab"] = ntab;
    res["rmax"] = rmax;
    res["dr"] = dr;
    res["op"] = std::string(1, layout.op_);
    return res;
}

torch::Tensor extract_gaunt_table(int lmax) {
    RealGauntTable::instance().build(lmax);
    int dim1 = (lmax + 1) * (lmax + 1);
    int dim2 = dim1;
    int dim3 = (2 * lmax + 1) * (2 * lmax + 1);
    auto g = torch::zeros({dim1, dim2, dim3}, torch::kFloat64);
    double* g_ptr = g.data_ptr<double>();
    
    for (int l1 = 0; l1 <= lmax; ++l1) {
        for (int m1 = -l1; m1 <= l1; ++m1) {
            int idx1 = l1 * (l1 + 1) + m1;
            for (int l2 = 0; l2 <= lmax; ++l2) {
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    int idx2 = l2 * (l2 + 1) + m2;
                    for (int l3 = std::abs(l1 - l2); l3 <= l1 + l2; l3 += 2) {
                        for (int m3 = -l3; m3 <= l3; ++m3) {
                            int idx3 = l3 * (l3 + 1) + m3;
                            double val = RealGauntTable::instance()(l1, l2, l3, m1, m2, m3);
                            g_ptr[idx1 * dim2 * dim3 + idx2 * dim3 + idx3] = val;
                        }
                    }
                }
            }
        }
    }
    return g;
}

} // namespace two_center