"""Version and trainable-parameter contracts."""

K_DEFAULT = 2
CORRECTNESS_VERSION = "loopscf-correctness-v2"
WM_KEYS = ("_wm_node_feat", "_wm_edge_feat")
ARM1_PATTERNS = (
    "embedding.out_node",
    "embedding.out_edge",
    "embedding.out_node_ele_tp",
    "embedding.out_edge_ele_tp",
    "wm_node",
    "wm_edge",
)
ARM2_PATTERNS = (
    "embedding.layers",
    "embedding.router",
    "embedding.out_node",
    "embedding.out_edge",
    "embedding.out_node_ele_tp",
    "embedding.out_edge_ele_tp",
    "wm_node",
    "wm_edge",
)
