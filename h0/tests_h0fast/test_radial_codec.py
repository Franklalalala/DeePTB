import numpy as np
import pytest
from h0rebuild.radial_codec import pack, unpack, payload_bytes


@pytest.mark.parametrize('segments',[1,2,127])
def test_arbitrary_coefficients_roundtrip_bits(segments):
    rng=np.random.default_rng(521)
    data=rng.normal(size=(4,segments,4))
    data[0,0,0]=-0.0;data[0,0,2]=-0.0
    encoded=pack(data,.01)
    assert np.array_equal(unpack(encoded).view(np.uint64),data.view(np.uint64))


def test_hermite_structure_shrinks_without_error():
    # Independent cubic polynomial sampled on a uniform grid.
    x=np.arange(1025,dtype=float)/128
    y=3+2*x+4*x*x+x*x*x;dy=2+8*x+3*x*x
    c=np.stack([y[:-1],dy[:-1],4+3*x[:-1],np.ones(1024)],axis=-1)[None]
    encoded=pack(c,1/128)
    assert np.array_equal(unpack(encoded).view(np.uint64),c.view(np.uint64))
    assert payload_bytes(encoded)<.51*c.nbytes


def test_invalid_input_rejected():
    with pytest.raises(ValueError):pack(np.ones((1,4,4),dtype=np.float32),.01)
    with pytest.raises(ValueError):pack(np.ones((1,4,4)),np.nan)
