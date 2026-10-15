"""Write-once table store: npz shards with their SHA256 under a root with its own manifest.

A store is one table family root (``<root>/manifest.json`` plus shard files at manifest-relative paths). Shards are
serialised in memory, hashed once and linked into place without ever replacing an existing file; the manifest is
rewritten atomically after each batch. Every manifest records the generator identity (``generator_identity``): the
SHA256 of the generating modules and the numerical environment, so a table can be traced to the code that wrote it.
"""
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def generator_identity(*modules):
    """{module: sha256 of its source} plus the numerical environment and device."""
    import torch
    out = {}
    for name in modules:
        mod = sys.modules.get(name) or __import__(name, fromlist=['_'])
        out[name] = sha256_file(mod.__file__)
    env = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
    if torch.cuda.is_available():
        env['device'] = torch.cuda.get_device_name()
    return dict(modules=out, environment=env)


def npz_bytes(arrays, compress=False):
    """An npz archive of ``arrays`` in memory (stored members unless ``compress``)."""
    buf = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, 'w', compression=mode, **({'compresslevel': 1} if compress else {})) as z:
        for key, value in arrays.items():
            with z.open(key + '.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)
    return buf.getbuffer()


class TableStore:
    """One family root. ``header`` fields are fixed when the root is created and must match when it is reopened."""

    def __init__(self, root, schema, **header):
        self.root = Path(root)
        self.path = self.root / 'manifest.json'
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text())
            if self.manifest.get('schema') != schema:
                raise ValueError(f'{root}: existing store has schema {self.manifest.get("schema")!r}, not {schema!r}')
            for k, v in header.items():
                if k in self.manifest and self.manifest[k] != v:
                    raise ValueError(f'{root}: existing store differs in {k}; use a new extension root')
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.manifest = dict(schema=schema, complete=True, **header)

    def has(self, section, key):
        return key in self.manifest.get(section, {})

    def generator(self, *modules):
        """Record the identity of the generating modules under ``generators`` -> its id (for table rows)."""
        ident = generator_identity(*modules)
        gid = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
        self.manifest.setdefault('generators', {})[gid] = ident
        return gid

    def write(self, relpath, data):
        """Place bytes at ``relpath`` (never replacing a file) -> (sha256, bytes)."""
        target = self.root / relpath
        if target.exists():
            raise FileExistsError(f'refusing to overwrite table file {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.partial-', delete=False) as f:
            f.write(data)
            temp = f.name
        try:
            os.link(temp, target)          # fails if the target appeared meanwhile
        finally:
            os.unlink(temp)
        return digest, len(data)

    def write_npz(self, relpath, arrays, compress=False):
        digest, size = self.write(relpath, npz_bytes(arrays, compress))
        return dict(path=str(relpath), sha256=digest, bytes=size)

    def add(self, section, key, row):
        rows = self.manifest.setdefault(section, {})
        if key in rows:
            raise FileExistsError(f'{self.root}: {section}[{key}] already exists')
        rows[key] = row

    def flush(self, **fields):
        self.manifest.update(fields, updated_unix=time.time())
        with tempfile.NamedTemporaryFile('w', dir=self.root, prefix='.manifest-', delete=False) as f:
            json.dump(self.manifest, f, indent=1, sort_keys=True)
            temp = f.name
        os.replace(temp, self.path)
