"""Call-scoped QEq geometry sharing, including concurrent CUDA consumers."""
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import pytest
import torch

from dptb.nn.heads.charge_response import (
    PeriodicGammaCache, periodic_gamma_scope, solve_periodic,
)


def geometry(device):
    cell = torch.tensor([[24., 0., 0.], [7., 22., 0.], [4., 5., 23.]],
                        dtype=torch.float64, device=device)
    gen = torch.Generator().manual_seed(7)
    pos = torch.rand(20, 3, generator=gen, dtype=torch.float64).to(device) @ cell
    return pos, cell


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream qualification")
@pytest.mark.parametrize("solver", ["dense", "ewald"])
def test_concurrent_heads_keep_separate_charge_solutions(solver):
    pos, cell = geometry("cuda")
    cache = PeriodicGammaCache()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    opts = dict(sigma=1.2, g_cut=4., max_modes=100000, solver=solver)

    def head(i):
        with torch.cuda.stream(streams[i]), torch.inference_mode(), periodic_gamma_scope(cache):
            chi = torch.arange(20, dtype=torch.float64, device="cuda")[:, None] * (i + 1)
            j = torch.full_like(chi, 8. + i)
            actual = solve_periodic(chi, j, pos.clone(), cell.clone(), **opts)
            # An explicit nested scope disables sharing for the independent reference.
            with periodic_gamma_scope(None):
                expected = solve_periodic(chi, j, pos, cell, **opts)
            ready = torch.cuda.Event()
            ready.record()
            return actual, expected, ready

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(head, (0, 1)))
    for actual, expected, ready in results:
        torch.cuda.current_stream().wait_event(ready)
        for a, b in zip(actual, expected):
            a.record_stream(torch.cuda.current_stream())
            b.record_stream(torch.cuda.current_stream())
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(results[0][0][0], results[1][0][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H200 numerical qualification")
def test_geometry_options_and_in_place_mutation_are_not_stale():
    pos, cell = geometry("cuda")
    cache = PeriodicGammaCache()
    opts = dict(sigma=1.2, g_cut=4., max_modes=100000, solver="ewald")
    with torch.inference_mode():
        cache.get(pos, cell, **opts)
        pos[2] += .3
        cell[1, 0] += .2
        actual = cache.get(pos, cell, **opts)
        expected = PeriodicGammaCache().get(pos, cell, **opts)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for changes in (dict(sigma=1.3), dict(g_cut=3.)):
            kw = dict(opts, **changes)
            actual = cache.get(pos, cell, **kw)
            expected = PeriodicGammaCache().get(pos, cell, **kw)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(ValueError):
            cache.get(pos, cell, **dict(opts, max_modes=1))


def test_scope_restores_on_exception_and_releases_owner():
    cache = PeriodicGammaCache()
    ref = weakref.ref(cache)
    with periodic_gamma_scope(None):
        with pytest.raises(RuntimeError, match="scope failure"):
            with periodic_gamma_scope(cache):
                raise RuntimeError("scope failure")
        with periodic_gamma_scope(None):
            pass
    del cache
    gc.collect()
    assert ref() is None
