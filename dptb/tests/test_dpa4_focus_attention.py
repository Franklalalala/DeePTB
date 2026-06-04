import torch
from e3nn import o3

from dptb.nn.embedding.lem_moe_v3 import (
    PostActivation0eFocusGate,
    SingleHead0eEnvelopeAttention,
)


def test_post_activation_0e_focus_gate_initializes_as_identity():
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    gate = PostActivation0eFocusGate(irreps, num_focus=2)

    message = torch.randn(5, irreps.dim)
    gated = gate(message)

    assert gated.shape == message.shape
    assert gate.focus_index.shape == (irreps.dim,)
    assert torch.allclose(gated, message)


def test_single_head_0e_attention_normalizes_per_destination_with_envelope():
    torch.manual_seed(1)
    attention = SingleHead0eEnvelopeAttention(
        node_scalar_dim=3,
        message_scalar_dim=3,
        latent_dim=4,
        attn_dim=5,
    )

    dst_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    node_scalars = torch.randn(3, 3)
    message_scalars = torch.randn(4, 3)
    latents = torch.randn(4, 4)
    cutoff = torch.tensor([1.0, 0.5, 1.0, 0.0])
    message = torch.randn(4, 7)

    weights = attention.attention_weights(
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=latents,
        cutoff_coeffs=cutoff,
        dim_size=3,
    )
    per_node = torch.zeros(3).scatter_add(0, dst_index, weights)

    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    assert weights[-1].item() == 0.0
    assert torch.all(per_node[:2] > 0)
    assert torch.all(per_node <= 1.0 + 1e-6)

    out = attention(
        message=message,
        dst_index=dst_index,
        node_scalars=node_scalars,
        message_scalars=message_scalars,
        latents=latents,
        cutoff_coeffs=cutoff,
        dim_size=3,
    )

    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[2], torch.zeros_like(out[2]))

