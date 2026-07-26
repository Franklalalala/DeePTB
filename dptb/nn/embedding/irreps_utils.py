"""Small shared helpers for maintained equivariant embeddings."""

import e3nn.o3 as o3


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Return whether a tensor product can produce ``ir_out``."""
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    return any(
        ir_out in ir1 * ir2
        for _, ir1 in irreps_in1
        for _, ir2 in irreps_in2
    )
