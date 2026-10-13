"""Explicit memory budgets, recorded in each acceptance execution identity."""
import math
import os


def memory_budgets():
    field = float(os.environ.get('H0_FIELD_MAX_WORK_MB', '4096'))
    sf = float(os.environ.get('H0_STRUCTURE_FACTOR_MAX_WORK_MB', '2048'))
    reuse = float(os.environ.get('H0_PROJECTOR_REUSE_MAX_MB', '0'))
    for name,value in (('H0_FIELD_MAX_WORK_MB',field),('H0_STRUCTURE_FACTOR_MAX_WORK_MB',sf)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be finite and positive')
    if not math.isfinite(reuse) or reuse < 0:
        raise ValueError('H0_PROJECTOR_REUSE_MAX_MB must be finite and non-negative')
    return {'field_max_work_mb': field, 'local_cache_max_mb': 4096.,
            'projector_reuse_max_mb': reuse,
            'structure_factor_max_work_mb': sf}
