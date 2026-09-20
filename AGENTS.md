# Maintained development branch

Use `0921-stable` as the single integration and maintenance branch for new DeePTB
features and fixes. Historical branches and immutable deployed releases are
compatibility references; do not continue separate feature release lines unless
the user explicitly requests one. Do not rewrite or delete historical branches
as part of ordinary integration work.

Keep one authoritative implementation per module. NACF lives in `dptb/nacf`;
the standalone H0 reconstruction subsystem lives in `h0`. Do not add review
overlays, copied source trees, production logs, checkpoints, or generated tables.

Use the focused behavioral testing policy in `TESTING.md`. Validate changed
interfaces and numerical behavior, and reuse relevant unchanged GPU evidence.
Keep scientific scope, prior/target semantics and checkpoint compatibility
explicit. Do not change or restart live production runs merely to update code.
