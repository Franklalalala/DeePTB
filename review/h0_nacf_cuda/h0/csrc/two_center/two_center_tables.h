#ifndef TWO_CENTER_TABLES_H_
#define TWO_CENTER_TABLES_H_

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <string>

#include "source_basis/module_nao/two_center_integrator.h"
#include "source_basis/module_nao/two_center_table.h"
#include "source_basis/module_nao/real_gaunt_table.h"

namespace two_center {

// Memory layout mirroring private members of TwoCenterTable in libnaopack
struct TwoCenterTableLayout {
    char op_;
    int ntab_;
    int nr_;
    double rmax_;
    double* rgrid_;
    container::Tensor nchi_ket_;
    container::Tensor table_;
    container::Tensor dtable_;
    container::Tensor index_map_;
};

// Offset of TwoCenterTable within TwoCenterIntegrator is 8 bytes
inline const TwoCenterTableLayout& get_table_layout(const TwoCenterIntegrator& intor) {
    return *reinterpret_cast<const TwoCenterTableLayout*>(reinterpret_cast<const char*>(&intor) + 8);
}

// Extract full table, derivative table, precomputed cubic polynomial coeffs and index_map
py::dict extract_radial_table(py::object intor_obj);

// Extract real Gaunt table tensor for given lmax
torch::Tensor extract_gaunt_table(int lmax);

} // namespace two_center

#endif // TWO_CENTER_TABLES_H_