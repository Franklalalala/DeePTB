# Optional zero-bias dpnegf transport bridge (M1)

`dptb.transport` is a narrow M1 integration layer for equilibrium,
zero-bias, frozen-H transmission. It accepts explicit dense device and lead
Hamiltonian/overlap matrices, validates their physical and indexing contract,
and evaluates the pinned dpnegf path
`surface_green.selfEnergy → recursive_green_cal.recursive_gf →
DeviceProperty._cal_tc_`.

This is a channel-count transmission bridge. It does not multiply the Caroli
transmission by the recorded spin degeneracy and it does not call the result a
current or a conductance.

## Installation

dpnegf is a runtime-optional dependency. DeePTB's `pyproject.toml` is
deliberately unchanged so that transport does not constrain the core
Torch/SciPy environment. Install the audited upstream commit in a dedicated
environment:

```powershell
python -m pip install `
  "git+https://github.com/deepmodeling/dpnegf.git@9b5da1296b7f0ca952b2e38742095d1d78b44434"
```

For an offline checkout at the audited commit:

```powershell
git -C C:\path\to\dpnegf rev-parse HEAD
# Must print 9b5da1296b7f0ca952b2e38742095d1d78b44434
python -m pip install C:\path\to\dpnegf
```

Importing `dptb.transport` does not import dpnegf. The first computation loads
only the three audited numerical modules and raises an actionable `RuntimeError`
if dpnegf is absent or its version does not contain pin `9b5da12`.

## Contract

Construct a `TransportConventions`, two `LeadPrincipalLayer` values, then a
`DenseHSProvider`. The provider requires:

- `complex128`, finite, k-resolved dense matrices;
- Hermitian device `H_D/S_D` and lead `H_00/S_00`;
- positive-definite device and principal-layer overlap matrices;
- shape-consistent `H_01/S_01` and device-lead `H_D0/S_D0` couplings;
- energy unit exactly `eV`, with required finite `E_ref`;
- explicit unique AO labels, atom-to-AO map, basis and m-order descriptions;
- explicit fractional-k convention and already normalized k weights;
- spin degeneracy `2` for `non-SOC spin-degenerate`, or `1` for
  `explicit-spin/SOC`;
- transport direction and the potential/energy convention
  `H(V)=H-V*S; E_abs=E_relative+E_ref`.

The M1 provider rejects every nonzero device or lead potential. This is an
intentional fail-closed boundary, not a finite-bias implementation.

```python
from dptb.transport import zero_bias_transmission

result = zero_bias_transmission(
    provider,
    energy_grid_eV_relative_to_E_ref,
    eta_lead=1.0e-5,
    eta_device=0.0,
)
```

`TransmissionResult` contains the k-averaged `transmission`, the
`transmission_k` array, and immutable provenance. Provenance records SHA-256
hashes of all 14 input matrix stacks, the full dpnegf commit and package
version, the energy grid, both eta values, and every AO/k/spin/E_ref
convention.

## Audited scope and risks

The integration rationale, upstream API map, M1 criteria, and risk table are
in `REPORT_D.md` sections 2.2, 8, 9, and 10 from the Wave-2 dpnegf audit. The
high-risk items addressed here are the absence of a stable public API,
undeclared or conflicting dependencies, mixed upstream version metadata, and
the need to isolate all upstream internals behind one adapter.

The pin and contract tests mitigate those risks; they do not make the upstream
Poisson/density/finite-bias paths M1-ready. dpnegf is LGPL-3.0 and remains a
separately installed dependency; no upstream source is vendored here.

## Explicitly out of scope

- finite-bias current or I-V integration;
- self-consistent bias-dependent `T(E,V)`;
- equilibrium or nonequilibrium density matrices;
- Poisson-NEGF or any claim that a finite-cluster electrostatic operator is a
  Poisson solver;
- the legacy dpnegf transport module, which is neither imported nor used;
- SEI leakage-current production claims.

Those capabilities require the later M2/M3 contracts, scientific
certificates, convergence checks, and bias-domain validation described in
`REPORT_D.md`.
