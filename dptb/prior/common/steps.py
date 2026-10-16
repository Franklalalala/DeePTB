"""Step timing and deferred checks of one prior call, the same rules for the GF and H0 priors.

Synchronisation: the device is synchronised only at step boundaries (``Steps``), so a step's wall-clock time includes
its kernels, and where a size read back from the device fixes an allocation. Checks and statistics are device scalars
collected during the call (``Checks``) and read back together once, at the end.
"""
import time

import torch


class Steps:
    """Wall-clock seconds per named step, ``<step>_s``, and ``total_s``."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.times = {}
        self.t0 = self.last = time.perf_counter()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).synchronize()

    def __call__(self, name):
        """End the current step as ``name``."""
        self.sync()
        now = time.perf_counter()
        self.add(name, now - self.last)
        self.last = now

    def add(self, name, seconds):
        self.times[name + '_s'] = self.times.get(name + '_s', 0.0) + seconds

    def done(self):
        self.times['total_s'] = time.perf_counter() - self.t0
        return self.times


class Checks:
    """Device-side failure flags and statistics, read back in one transfer by ``read``."""

    def __init__(self):
        self._fail, self._values = [], {}

    def fail_if(self, flag, message):
        """Fail with ``message`` when the device scalar ``flag`` is nonzero."""
        self._fail.append((flag, message))

    def value(self, name, x):
        """Record the device scalar ``x`` as statistic ``name``."""
        self._values[name] = x

    def read(self):
        """One device-to-host transfer: raises ValueError for the first failed check, else returns the statistics."""
        names = list(self._values)
        xs = [f for f, _ in self._fail] + [self._values[k] for k in names]
        if not xs:
            return {}
        host = torch.stack([torch.as_tensor(x).reshape(()).to(torch.float64) for x in xs]).tolist()
        for (_, message), h in zip(self._fail, host):
            if h:
                raise ValueError(message)
        return dict(zip(names, host[len(self._fail):]))
