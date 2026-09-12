# Source snapshot scope

`h0/` and `nacf_overlay/` preserve the task's reviewed code under a review-only directory.
The existing repository production tree is unchanged. This is not an installable merged release.
The NACF overlay is based on the task's SOC integration source; do not overlay it blindly onto
an arbitrary DeePTB revision. The local handoff identifies that exact parent and installed environment.
The focused NACF patch isolates this task's seven runtime/cache changes from inherited SOC code.

GitHub contains source, tests, focused diffs, checksums and aggregate review context. It contains no
credentials, raw H/S/UPF/ORB datasets, paused process inventory, installed 8.9 GiB tables or binaries.
The separate local handoff includes numerical result records and ABI-specific prebuilt artifacts.
It is not necessary to rebuild the existing installed binaries to review or resume that environment.
