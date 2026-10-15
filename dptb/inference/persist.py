"""Optional asynchronous, uncompressed persistence of block tables (layout compact-blocks-v1).

A table is packed on the device: keys sorted by (i, j, R) (int32 [n, 5]), block shapes (int16 [n, 2]), offsets
(int64 [n + 1]) and the blocks [:rows, :cols] raveled in C order (one flat array). A writer thread copies the packed
arrays to the host on its own CUDA stream, after an event recorded behind the packing kernels, so the copy overlaps
the next structure; it writes one HDF5 file per table (datasets values, keys, shapes, offsets; attributes schema,
quantity, units) and renames it into place. At most ``depth`` structures wait for the writer.
"""
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import torch

from dptb.prior.common.neighbors import key_codes, span_of

SCHEMA = 'compact-blocks-v1'


def pack(table, n_atoms, dtype=None):
    """BlockTable -> (keys int32 [n, 5], shapes int16 [n, 2], offsets int64 [n + 1], flat values) on the device."""
    keys = table.keys
    order = torch.argsort(key_codes(keys[:, 0], keys[:, 1], keys[:, 2:], n_atoms, span_of(keys[:, 2:])))
    rows, cols = table.rows[order], table.cols[order]
    w1, w2 = table.values.shape[1:]
    parts = []
    for a in range(0, len(order), 65536):           # bounded transient: one chunk of padded blocks at a time
        o, r, c = order[a:a + 65536], rows[a:a + 65536], cols[a:a + 65536]
        m = (torch.arange(w1, device=r.device)[None, :, None] < r[:, None, None]) & \
            (torch.arange(w2, device=r.device)[None, None, :] < c[:, None, None])
        parts.append(table.values[o][m])
    values = torch.cat(parts) if parts else table.values.new_zeros(0)
    if dtype is not None:
        values = values.to(dtype)
    sizes = rows.to(torch.int64) * cols.to(torch.int64)
    offsets = torch.cat([sizes.new_zeros(1), torch.cumsum(sizes, 0)])
    return keys[order].to(torch.int32), torch.stack([rows, cols], 1).to(torch.int16), offsets, values


def write(path, quantity, keys, shapes, offsets, values):
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with h5py.File(tmp, 'w') as f:
        f.attrs.update(schema=SCHEMA, quantity=quantity, units='1' if quantity == 'S' else 'eV')
        for name, data in (('values', values), ('keys', keys), ('shapes', shapes), ('offsets', offsets)):
            f.create_dataset(name, data=data)
    os.replace(tmp, path)
    return dict(path=str(path), bytes=path.stat().st_size, blocks=int(len(keys)))


class Writer:
    def __init__(self, depth=2):
        self.q = queue.Queue(maxsize=max(1, depth))
        self.done, self.errors = [], []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        stream = torch.cuda.Stream()
        while True:
            job = self.q.get()
            if job is None:
                return
            t = time.perf_counter()
            try:
                with torch.cuda.stream(stream):
                    stream.wait_event(job['ready'])
                    host = [(path, quantity) + tuple(a.cpu().numpy() for a in arrays)
                            for path, quantity, arrays in job['files']]
                job['files'] = None
                files = [write(*f) for f in host]
                self.done.append(dict(tag=job['tag'], files=files, write_s=time.perf_counter() - t))
            except Exception as exc:      # reported by drain(); the compute loop keeps going
                self.errors.append(dict(tag=job['tag'], error=f'{type(exc).__name__}: {exc}'))
            finally:
                self.q.task_done()

    def submit(self, tag, tables, n_atoms):
        """tables: [(path, quantity, BlockTable, dtype or None)]; returns after packing (the copy is asynchronous)."""
        files = [(path, quantity, pack(table, n_atoms, dtype)) for path, quantity, table, dtype in tables]
        ready = torch.cuda.Event()
        ready.record()
        self.q.put(dict(tag=tag, files=files, ready=ready))

    def drain(self):
        self.q.join()
        if self.errors:
            raise RuntimeError('persistence failed: %s' % self.errors)
        return self.done

    def close(self):
        self.drain()
        self.q.put(None)
        self.thread.join()
