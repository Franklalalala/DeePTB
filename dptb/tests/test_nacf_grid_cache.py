"""Grid support pruning must follow supported changes of the quadrature."""
from types import SimpleNamespace
import torch
from dptb.nacf.onsite import grid_radius, invalidate_grid_radius


def test_grid_cache_tracks_replacement_inplace_and_view_shape():
    q = SimpleNamespace(xyz=torch.zeros((2, 3), dtype=torch.float64))
    assert grid_radius(q) == 0
    q.xyz = torch.tensor([[3., 4., 0.], [0., 0., 0.]], dtype=torch.float64)
    assert grid_radius(q) == 5
    q.xyz.zero_()
    assert grid_radius(q) == 0
    q.xyz[1, 0] = 7
    assert grid_radius(q) == 7
    q.xyz = q.xyz[:1]
    assert grid_radius(q) == 0


def test_untracked_grid_edits_have_explicit_invalidation():
    with torch.inference_mode():
        q = SimpleNamespace(xyz=torch.zeros((1, 3), dtype=torch.float64))
        assert grid_radius(q) == 0
        q.xyz[0, 0] = 5
        invalidate_grid_radius(q)
        assert grid_radius(q) == 5
    q = SimpleNamespace(xyz=torch.zeros((1, 3), dtype=torch.float64))
    assert grid_radius(q) == 0
    q.xyz.numpy()[0, 1] = 4
    invalidate_grid_radius(q)
    assert grid_radius(q) == 4
