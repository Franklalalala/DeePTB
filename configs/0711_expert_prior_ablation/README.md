# 0711 expert prior ablation

These four runs extend the recovered A/B expert-DDP comparison while keeping
the same dataset size, model, seed, optimizer, two-expert onsite/hopping split,
and 10,000-iteration budget.

| Run | Objective | Native node InitLayer | Node physical input | Edge physical input |
| --- | --- | --- | --- | --- |
| C CFM | CFM | replaced | B: `node_overlap` | A: `edge_h0` |
| D CFM | CFM | retained | none | A: `edge_h0` |
| C non-CFM | direct `hamil_abs` | replaced | B: `node_overlap` | A: `edge_h0` |
| D non-CFM | direct `hamil_abs` | retained | none | A: `edge_h0` |
| E baseline | direct `hamil_abs` | replaced | A: `node_h0` | A: `edge_h0` |

For C CFM, `prior_node=external` and `prior_edge=external` select different
absolute fields through `prior_node_key` and `prior_edge_key`. Since the edge
field equals the residual-mode H0 base, its flow residual prior is exactly
zero, reproducing A on the edge side.

For D, `model_options.embedding.use_h0_node_init=false` preserves the native
NN InitLayer node output while `use_h0_edge_init=true` retains H0 edge
initialization. The non-CFM variants keep the same static initialization but
set `train_options.flow_options.enabled=false`; they therefore remain distinct
instead of collapsing to one ordinary baseline.

The E baseline disables flow but keeps the complete H0 InitLayer enabled, so
both node and edge use A's H0 initialization under the direct `hamil_abs` loss.
