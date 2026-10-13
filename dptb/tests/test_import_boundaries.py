"""Model imports preserve application logging and data workers own field contracts."""
from pathlib import Path
import os
import subprocess
import sys


def _isolated(code):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1"),
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prior_baseline_import_preserves_caller_logging():
    _isolated('''
import importlib
import io
import logging

class CallerHandler(logging.StreamHandler):
    was_closed = False
    def close(self):
        self.was_closed = True
        super().close()

stream = io.StringIO()
handler = CallerHandler(stream)
formatter = logging.Formatter('caller:%(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.WARNING)
root = logging.getLogger()
root.handlers[:] = [handler]
root.setLevel(logging.ERROR)

importlib.import_module('dptb.nn.embedding.lem_prior')

assert root.handlers == [handler]
assert root.level == logging.ERROR
assert handler.level == logging.WARNING
assert handler.formatter is formatter
assert not handler.was_closed
root.error('after import')
assert stream.getvalue() == 'caller:after import\\n'
''')


def test_residual_field_alignment_and_collation_work_in_a_data_only_process():
    _isolated('''
import sys
import torch
from dptb.data import AtomicData, Collater, _keys

node_fields = (
    _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
    _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
    _keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
)
edge_fields = (
    _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
    _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
    _keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
)

def assert_data_only():
    assert not any(name.startswith(('dptb.nn.embedding.', 'dptb.nn.conditioning.'))
                   for name in sys.modules)

assert_data_only()
graphs = []
for size in (2, 3):
    edge_index = torch.stack((torch.arange(size - 1), torch.arange(1, size)))
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    values = {}
    for field in node_fields + edge_fields:
        count = size if field in node_fields else edge_index.shape[1]
        if field.endswith('_shape'):
            values[field] = torch.full((count, 2), 2, dtype=torch.long)
        else:
            values[field] = torch.arange(count * 4).float().reshape(count, 2, 2)
    graphs.append(AtomicData(pos=torch.zeros(size, 3), edge_index=edge_index, **values))

batch = Collater()(graphs)
for field in node_fields + edge_fields:
    assert torch.equal(batch[field], torch.cat([graph[field] for graph in graphs]))
    for index, graph in enumerate(graphs):
        assert torch.equal(batch.get_example(index)[field], graph[field])

for field in node_fields + edge_fields:
    values = AtomicData.to_AtomicDataDict(graphs[1])
    values[field] = torch.cat((values[field], values[field][:1]), dim=0)
    try:
        AtomicData(**values)
    except ValueError:
        pass
    else:
        raise AssertionError('misaligned residual field was accepted: ' + field)

assert_data_only()
''')
