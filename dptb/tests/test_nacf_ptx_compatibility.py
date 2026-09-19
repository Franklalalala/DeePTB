"""Recorded PTX permits new GPUs without weakening source/binary checks."""
import json
import pytest
from dptb.nacf import precompiled


@pytest.mark.parametrize('cap,ptx,accepted', [
    ((8,9), [], True), ((12,0), [], False),
    ((12,0), ['8.9'], True), ((8,6), ['8.9'], False),
])
def test_recorded_ptx_compatibility(tmp_path, monkeypatch, cap, ptx, accepted):
    binary=tmp_path/'test.so';binary.write_bytes(b'fixture')
    m={'runtime':precompiled.identity(),'architectures':['8.9'],
       'ptx_architectures':ptx,'sources':{},'binary':binary.name,
       'sha256':precompiled.sha(binary)}
    (tmp_path/'prebuilt.json').write_text(json.dumps(m))
    monkeypatch.setattr(precompiled,'ROOT',tmp_path)
    monkeypatch.setattr(precompiled.torch.cuda,'get_device_capability',lambda device:cap)
    if accepted:
        assert precompiled.verify('cuda')['sha256']==m['sha256']
        binary.write_bytes(b'corrupt')
        with pytest.raises(RuntimeError,match='checksum'):precompiled.verify('cuda')
    else:
        with pytest.raises(RuntimeError,match='architecture'):precompiled.verify('cuda')
