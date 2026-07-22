# Multiplicity-tied irrep Gaussian prior

This note defines the `tied_irrep_gaussian` prior by its coordinate-space and
representation semantics.  It also gives a fixed numerical example so that the
symbols used by the implementation can be followed from a Gaussian latent to a
physical Hamiltonian block.

## 1. Coordinate spaces and symbols

The prior is used only by the non-SOC direct-residual block flow.  Its endpoint
is

\[
D_1 = H - H_0,
\]

where `H` is the reference full Hamiltonian and `H0` is the fixed physical
baseline.  The model state is the residual `D`, not the full Hamiltonian.

| Symbol | Space | Meaning |
|---|---|---|
| \(z\) | abstract irrep latent | A conceptual Gaussian draw before multiplicity tying. It is not an AO matrix and is not an `H0` feature. |
| \(g_0,g_1,g_2\) | effective SO(3) irrep latent | The scalar, vector, and rank-2 fields remaining after equal-weight multiplicity copies are summed. |
| \(\epsilon_{\mathrm{RME}}\) | node/edge RME rows | Reduced-matrix-element coefficients obtained by broadcasting the same \(g_L\) into all active copies with total angular degree \(L\). |
| \(\epsilon_{\mathrm{block}}\) | onsite/hopping AO block canvas | The RME prior after the fixed CG expansion and physical block projection. |
| \(\epsilon_{ss}\) | sub-block of \(\epsilon_{\mathrm{block}}\) | The rows and columns belonging to the selected `s` radial shells. It is not a separate draw. |
| \(D_t\) | residual AO block state | The interpolated state \((1-t)\epsilon_{\mathrm{block}}+tD_1\), projected onto the certified block-state space. |

The corresponding full-H state is only an interpretation:

\[
H_t = H_0 + D_t.
\]

`H0` remains a separate, constant conditioning channel throughout the ODE.  It
is added to the final residual exactly once when exporting full `H`.

## 2. What the Gaussian latent contains

For the water basis used by the reference configuration, the conceptual latent
representation is

\[
z = 3\times 0e \;\oplus\;2\times 1e\;\oplus\;1\times 2e.
\]

It contains

\[
z=(a_1,a_2,a_3)\oplus(b_1,b_2)\oplus c,
\]

where each \(a_r\) is a scalar, each \(b_r\) has three magnetic components,
and \(c\) has five magnetic components.  The raw dimension is therefore

\[
3\times1+2\times3+1\times5=14.
\]

Equal weights are used across the multiplicity copies.  Consequently, only

\[
g_0=a_1+a_2+a_3,\qquad
g_{1m}=b_{1m}+b_{2m},\qquad
g_{2m}=c_m
\]

affect the expanded state.  Rather than draw 14 values and immediately sum
them, the implementation draws the distribution-equivalent effective fields

\[
g_0\sim\mathcal N(0,3),\qquad
g_{1m}\sim\mathcal N(0,2),\qquad
g_{2m}\sim\mathcal N(0,1).
\]

There are therefore at most

\[
1+3+5=9
\]

effective random directions per node or edge row before masks and physical
projection.  This is the sense in which the prior is low-dimensional.  It does
not mean that a generated AO matrix must have numerical matrix rank nine.

## 3. How an irrep latent becomes an AO block

Let orbital shell \(\alpha\) have angular momentum \(l_\alpha\) and shell
\(\beta\) have angular momentum \(l_\beta\).  Their product decomposes as

\[
l_\alpha\otimes l_\beta
=|l_\alpha-l_\beta|\oplus\cdots\oplus(l_\alpha+l_\beta).
\]

For every allowed total degree \(L\), the fixed CG transform expands an RME
coefficient into magnetic AO entries:

\[
\epsilon_{\alpha m_\alpha,\,\beta m_\beta}
=\sum_{L,M}
C^{LM}_{l_\alpha m_\alpha,\,l_\beta m_\beta}
\;r_{\alpha\beta,LM}.
\]

The tied prior sets all active copies with the same total degree to the same
effective field, up to the fixed convention and mask:

\[
r_{\alpha\beta,LM}=\sigma\,g_{LM}.
\]

This is a linear CG expansion.  It is **not** scalar gating: `0e` is not used as
a scale multiplying the `1e` or `2e` fields.

The shell-pair selection rules make this concrete:

| Shell pair | Product | Which effective latent can contribute? |
|---|---|---|
| `s-s` | \(0\otimes0=0\) | only \(g_0\) |
| `s-p` | \(0\otimes1=1\) | only \(g_1\) |
| `s-d` | \(0\otimes2=2\) | only \(g_2\) |
| `p-p` | \(1\otimes1=0\oplus1\oplus2\) | \(g_0,g_1,g_2\) |
| `p-d` | \(1\otimes2=1\oplus2\oplus3\) | \(g_1,g_2\); the configured prior has no \(g_3\) |
| `d-d` | \(2\otimes2=0\oplus1\oplus2\oplus3\oplus4\) | \(g_0,g_1,g_2\); \(L=3,4\) are zero |

In particular, `s1-p1` comes from the effective `1e` vector because

\[
0\otimes1=1
\]

contains no \(L=0\) or \(L=2\) channel.  A scalar can change an `s-p` block
only in a different architecture that explicitly multiplies a scalar gate by
a vector.  This prior does not perform such a multiplication.

## 4. Fixed numerical expansion example

Consider the following fixed conceptual 14-component vector:

```text
z = [
   0.10, -0.20,  0.30,                 # a1, a2, a3: 3 x 0e
   0.01,  0.02,  0.03,                 # b1:          first 1e copy
  -0.04,  0.05, -0.06,                 # b2:          second 1e copy
   0.07, -0.08,  0.09, -0.10, 0.11    # c:           1 x 2e
]
```

Its effective fields are

```text
g0 = 0.10 - 0.20 + 0.30 = 0.20
g1 = (0.01, 0.02, 0.03) + (-0.04, 0.05, -0.06)
   = (-0.03, 0.07, -0.03)
g2 = (0.07, -0.08, 0.09, -0.10, 0.11)
```

Using the e3nn real-spherical Wigner-3j convention and equal multiplicity
weights, the three-by-three block among the three `s` radial shells is

```text
epsilon_ss =
[[0.2000, 0.2000, 0.2000],
 [0.2000, 0.2000, 0.2000],
 [0.2000, 0.2000, 0.2000]]
```

All nine values are equal because `s-s` admits only \(L=0\), and every radial
copy receives the same `g0`.  The nine values are not nine independent noise
draws.

For the first `p` radial shell, the raw three-by-three expansion is

```text
epsilon_p1p1_raw =
[[ 0.0009, -0.0778,  0.0000],
 [-0.0354,  0.1890, -0.0919],
 [ 0.0990, -0.0495,  0.1565]]
```

The `p-p` product contains \(L=0,1,2\).  Its \(L=1\) component is
antisymmetric.  A legal real onsite Hamiltonian is symmetric, so the physical
block projection removes that antisymmetric component:

```text
epsilon_p1p1 = 0.5 * (epsilon_raw + epsilon_raw.T)

             =
[[ 0.0009, -0.0566,  0.0495],
 [-0.0566,  0.1890, -0.0707],
 [ 0.0495, -0.0707,  0.1565]]
```

The same effective vector and rank-2 field simultaneously produce

```text
epsilon_s1p1 =
[[-0.0300, 0.0700, -0.0300]]

epsilon_s1d =
[[0.0700, -0.0800, 0.0900, -0.1000, 0.1100]]
```

`epsilon_s1p1` is generated from `g1`; `epsilon_s1d` is generated from `g2`.
Neither is obtained by multiplying the scalar `g0` into another irrep.

All five numbers above are printed directly by the fixed
`dense_all_one_irrep_expansion` (a throwaway script, not hand arithmetic) and
independently cross-checked bit-exact against the real production codec
(`fill_tied_irrep_rme` -> `flow.block_codec.rme_to_blocks`, i.e.
`E3Hamiltonian`) on water's real oxygen `3s2p1d` row -- see
`test_dense_all_one_expansion_matches_production_codec_on_water_oxygen_row`
in `dptb/tests/test_tied_irrep_gaussian_prior.py`.  An earlier version of
this section was generated from a `dense_all_one_irrep_expansion` that was
missing the standard `sqrt(2L+1)` Wigner-3j-to-Clebsch-Gordan normalization
factor for every `L>=1` channel; `epsilon_s1p1`/`epsilon_s1d` were off by
`sqrt(3)`/`sqrt(5)` and `epsilon_p1p1`'s non-scalar channels were mis-scaled
by different factors (only its trace, the pure `L=0` part, was correct). The
`epsilon_ss` block was unaffected (`L=0` only, `sqrt(1)=1`).

## 5. Before and after adding the prior

Take the following illustrative physical `p1-p1` block of `H0`, in the native
Hamiltonian energy unit:

```text
H0 =
[[-0.5000,  0.0200,  0.0000],
 [ 0.0200, -0.4800,  0.0100],
 [ 0.0000,  0.0100, -0.5200]]
```

With the exact-zero prior,

```text
D0 = 0
full-state interpretation at t=0: H0 + D0 = H0
```

With the tied-irrep prior and `sigma=1`,

```text
D0 = epsilon_p1p1

H0 + D0 =
[[-0.4991, -0.0366,  0.0495],
 [-0.0366, -0.2910, -0.0607],
 [ 0.0495, -0.0607, -0.3635]]
```

The model receives `D0` as its evolving residual block state and receives `H0`
separately as the fixed physical condition.

Now choose an illustrative target residual

```text
D1 = H - H0 =
[[ 0.0200,  0.0100, 0.0000],
 [ 0.0100, -0.0100, 0.0050],
 [ 0.0000,  0.0050, 0.0300]]
```

At `t=0.25`,

\[
D_{0.25}=0.75\epsilon+0.25D_1,
\]

which gives

```text
D_0.25 =
[[ 0.0057, -0.0399,  0.0371],
 [-0.0399,  0.1392, -0.0518],
 [ 0.0371, -0.0518,  0.1249]]

H0 + D_0.25 =
[[-0.4943, -0.0199,  0.0371],
 [-0.0199, -0.3408, -0.0418],
 [ 0.0371, -0.0418, -0.3951]]
```

At `t=1`, the prior contribution is zero:

```text
H0 + D1 =
[[-0.4800,  0.0300, 0.0000],
 [ 0.0300, -0.4900, 0.0150],
 [ 0.0000,  0.0150,-0.4900]]
```

The endpoint language is therefore

```text
t=0:   pure tied-irrep residual prior D0 = epsilon
t=0.5: D = 0.5 * epsilon + 0.5 * (H - H0)
t=1:   pure reference residual D1 = H - H0
```

## 6. Runtime path

```text
one independent rowwise draw for each node and edge
    |
    +-- g0 ~ Normal(0, 3)
    +-- g1[m] ~ Normal(0, 2), m = -1, 0, 1
    `-- g2[m] ~ Normal(0, 1), m = -2, ..., 2
    |
    v
broadcast gL into every active RME copy with the same total degree L
    |
    +-- L = 0, 1, 2 are populated
    `-- L >= 3 are exactly zero
    |
    v
apply atom/edge-type RME masks
    |
    v
fixed RME-to-AO CG expansion
    |
    v
physical block projection
    +-- onsite Hermiticity
    +-- reverse-edge transpose pairing
    `-- zero padding outside species shapes
    |
    v
certified residual prior epsilon_block
    |
    v
D_t = project((1-t) * epsilon_block + t * (H-H0))
    |
    v
model(D_t, H0, t) -> predicted residual endpoint
```

Seeded validation uses stable per-sample substreams.  Training uses fresh draws,
but training and sampling call the same prior constructor and therefore share
the same distribution and projection rules.

## 7. Configuration semantics

The prior is selected explicitly:

```yaml
train_options:
  flow_options:
    output_space: residual_ao_block_ode
    state_space: residual_ao_block
    target_semantics: residual_dh
    prior: tied_irrep_gaussian
    tied_irrep_mode: so3_tied
    tied_irrep_irreps: "3x0e + 2x1e + 1x2e"
    tied_irrep_sigma: 1.0
    tied_irrep_validation_seed: 20260721
```

`tied_irrep_sigma` is a global multiplier applied after the multiplicity
variances `3/2/1` have been established.  The prior is not scaled from the
reference residual or from batch statistics.

## 8. Meaning of "rank"

Several unrelated uses of `rank` must not be conflated:

| Term | Meaning |
|---|---|
| irrep degree \(L=0,1,2\) | Angular-momentum transformation type. |
| effective prior rank at most 9 | Dimension of the rowwise Gaussian support before masks/projection. |
| covariance rank | Rank of \(P\operatorname{Cov}(g)P^T\) after the fixed latent-to-RME map. |
| decoder dynamic rank | A separate output-head parameterization choice. |
| numerical AO-matrix rank | Rank of one particular generated matrix; it need not equal the prior support rank. |

For example, an O--O RME row can have 121 active coordinates while all of them
are correlated functions of at most nine effective Gaussian variables.

## 9. Deliberate scope and limitations

- `so3_tied` reproduces rotation-degree tying.  Sharing one field across output
  copies of different inversion parity is not claimed to be an exact O(3)
  joint-distribution construction.
- The certified block projection changes the raw unprojected covariance by
  enforcing onsite and reverse-edge Hamiltonian constraints.  This is required
  by the block-ODE state contract.
- The prior is supported only for the non-SOC direct-residual block state.
- It does not add `H0` to the prior, does not read the target to choose its
  scale, and does not populate total degrees above \(L=2\).
