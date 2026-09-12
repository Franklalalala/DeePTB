# Scope of this continuation

Changes are confined to review/h0_nacf_cuda. The original reviewed parent is
72f3cb6f51fdee54143847982665d203194eea0d. No production model/training tree is changed.
The NACF overlay still requires its original SOC integration parent; do not copy
it blindly into another checkout. This continuation preserves prior/residual,
P23-onsite/P2-edge, unit and spin conventions.

The historical patches against Gemini and the SOC parent are retained as evidence
of the preceding snapshot. Use the Git diff against 72f3cb6 for this repair set.
The installed H0 runtime was frozen before full-cohort dispatch; repository-only
review documentation and test additions do not alter its computational identity.
Environment-specific prebuilt binaries and preparation/runtime logs are kept in
the local/remote task workspaces, outside Git. Large tables remain on Liyue.
