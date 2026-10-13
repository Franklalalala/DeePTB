# Output-route configuration snippets

Each `route_*.yaml` file is a minimal configuration for one registered output
route (RME, late-CG and AO-projector heads). They are used by
`dptb/tests/test_output_routes.py` and show the matching embedding, prediction
head and output space. Merge a snippet into a complete input with real basis,
data, prior and training settings before use.
