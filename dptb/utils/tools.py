import os
import numpy as np
import torch
from dptb.utils.constants import atomic_num_dict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import json
from pathlib import Path
import yaml
import torch.optim as optim
import logging
import random
from dptb.utils.dpa4_optim import HybridMuon, WarmupStableDecayLR, WarmupThenReduceLROnPlateau
import ssl
import os.path as osp
import urllib
import zipfile
import sys


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal["relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"]
    _PRECISION = Literal["default", "float16", "float32", "float64"]


def _strip_named_optimizer_params(model_param):
    def strip_one(item):
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and torch.is_tensor(item[1])
        ):
            return item[1]
        return item

    def strip_sequence(seq):
        if torch.is_tensor(seq):
            return seq
        return [strip_one(item) for item in list(seq)]

    if torch.is_tensor(model_param):
        return model_param
    if isinstance(model_param, dict):
        group = dict(model_param)
        group["params"] = strip_sequence(group["params"])
        return [group]

    items = list(model_param)
    if all(isinstance(item, dict) for item in items):
        groups = []
        for item in items:
            group = dict(item)
            group["params"] = strip_sequence(group["params"])
            groups.append(group)
        return groups
    return [strip_one(item) for item in items]


def float2comlex(dtype):
    if isinstance(dtype, str):
        dtype =  getattr(torch, dtype)
    
    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        raise ValueError("the dtype is not supported! now only float64, float32 is supported!")
    return cdtype


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_optimizer(type: str, model_param, lr: float, **options: dict):
    if type == 'Adam':
        model_param = _strip_named_optimizer_params(model_param)
        optimizer = optim.Adam(params=model_param, lr=lr, **options)
    elif type == 'AdamW':
        model_param = _strip_named_optimizer_params(model_param)
        optimizer = optim.AdamW(params=model_param, lr=lr, **options)
    elif type == 'HybridMuon':
        optimizer = HybridMuon(params=model_param, lr=lr, **options)
        summary = optimizer.route_summary()
        log.info(
            "HybridMuon routes: params_muon=%s/%s, flat_1d_muon=%s, "
            "numel_muon=%s/%s (%.2f%%), flat_numel=%s (%.2f%%)",
            summary["params_muon"],
            summary["params_total"],
            summary["params_1d_muon"],
            summary["numel_muon"],
            summary["numel_total"],
            100.0 * summary["numel_muon_ratio"],
            summary["numel_flat_muon"],
            100.0 * summary["numel_flat_muon_ratio"],
        )
    elif type == 'SGD':
        model_param = _strip_named_optimizer_params(model_param)
        optimizer = optim.SGD(params=model_param, lr=lr, **options)
    elif type == 'RMSprop':
        model_param = _strip_named_optimizer_params(model_param)
        optimizer = optim.RMSprop(params=model_param, lr=lr, **options)
    elif type == 'LBFGS':
        model_param = _strip_named_optimizer_params(model_param)
        optimizer = optim.LBFGS(params=model_param, lr=lr, **options)
    else:
        raise RuntimeError("Optimizer should be Adam/AdamW/HybridMuon/SGD/RMSprop, not {}".format(type))
    return optimizer

def get_lr_scheduler(type: str, optimizer: optim.Optimizer, **sch_options):
    if type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **sch_options)
    elif type == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, **sch_options)
    elif type == "rop":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **sch_options)
    elif type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **sch_options)
    elif type == 'wsd':
        scheduler = WarmupStableDecayLR(optimizer=optimizer, **sch_options)
    elif type == "warmup_rop":
        scheduler = WarmupThenReduceLROnPlateau(optimizer=optimizer, **sch_options)
    elif type == "cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, **sch_options)
    elif type == "qhflow_poly":
        scheduler = _make_qhflow_poly_lr_scheduler(optimizer=optimizer, **sch_options)
    else:
        raise RuntimeError("Scheduler should be exp/linear/rop/cos/wsd/warmup_rop/cyclic/qhflow_poly..., not {}".format(type))

    return scheduler


def _make_qhflow_poly_lr_scheduler(
    optimizer: optim.Optimizer,
    *,
    warmup_step: int = 1000,
    num_training_steps: int = 200000,
    end_lr: float = 1.0e-9,
    scheduler_power: float = 1.0,
    last_epoch: int = -1,
):
    warmup_step = int(warmup_step)
    num_training_steps = int(num_training_steps)
    end_lr = float(end_lr)
    scheduler_power = float(scheduler_power)
    if warmup_step < 0:
        raise ValueError("warmup_step must be >= 0 for qhflow_poly scheduler")
    if num_training_steps <= warmup_step:
        raise ValueError("num_training_steps must be larger than warmup_step for qhflow_poly scheduler")
    if end_lr < 0:
        raise ValueError("end_lr must be >= 0 for qhflow_poly scheduler")
    if scheduler_power <= 0:
        raise ValueError("scheduler_power must be > 0 for qhflow_poly scheduler")

    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    end_factors = [
        0.0 if base_lr <= 0 else min(max(end_lr / base_lr, 0.0), 1.0)
        for base_lr in base_lrs
    ]

    def lr_factor(step: int, end_factor: float):
        step = max(0, int(step))
        if warmup_step > 0 and step < warmup_step:
            return max(float(step) / float(warmup_step), 1.0e-12)
        progress = (step - warmup_step) / float(num_training_steps - warmup_step)
        progress = min(max(progress, 0.0), 1.0)
        decay = (1.0 - progress) ** scheduler_power
        return end_factor + (1.0 - end_factor) * decay

    lr_lambdas = [
        (lambda step, end_factor=end_factor: lr_factor(step, end_factor))
        for end_factor in end_factors
    ]
    return optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambdas, last_epoch=last_epoch)


def lr_scheduler_requires_metric(scheduler) -> bool:
    return (
        isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
        or bool(getattr(scheduler, "requires_metric", False))
    )


def lr_scheduler_can_step_without_metric(scheduler) -> bool:
    can_step = getattr(scheduler, "can_step_without_metric", None)
    return bool(callable(can_step) and can_step())

def j_must_have(
    jdata: Dict[str, "_DICT_VAL"], key: str, deprecated_key: List[str] = []
) -> "_DICT_VAL":
    """Assert that supplied dictionary conaines specified key.

    Returns
    -------
    _DICT_VAL
        value that was store unde supplied key

    Raises
    ------
    RuntimeError
        if the key is not present
    """
    if key not in jdata.keys():
        for ii in deprecated_key:
            if ii in jdata.keys():
                log.warning(f"the key {ii} is deprecated, please use {key} instead")
                return jdata[ii]
        else:
            raise RuntimeError(f"json database must provide key {key}")
    else:
        return jdata[key]

def get_uniq_symbol(atomsymbols):
    '''>It takes a list of atomic symbols and returns a list of unique atomic symbols in the order of
    atomic number
    
    Parameters
    ----------
    atomsymbols
        a list of atomic symbols, e.g. ['C', 'C','H','H',...]
    
    Returns
    -------
        the unique atom types in the system, and the types are sorted descending order of atomic number.
    
    '''
    atomic_num_dict_r = dict(zip(atomic_num_dict.values(), atomic_num_dict.keys()))
    atom_num = []
    for it in atomsymbols:
        atom_num.append(atomic_num_dict[it])
    # uniq and sort.
    uniq_atom_num = sorted(np.unique(atom_num), reverse=True)
    # assert(len(uniq_atom_num) == len(atomsymbols))
    uniqatomtype = []
    for ia in uniq_atom_num:
        uniqatomtype.append(atomic_num_dict_r[ia])

    return uniqatomtype

def j_loader(filename: Union[str, Path]) -> Dict[str, Any]:
    """Load yaml or json settings file.

    Parameters
    ----------
    filename : Union[str, Path]
        path to file

    Returns
    -------
    Dict[str, Any]
        loaded dictionary

    Raises
    ------
    TypeError
        if the supplied file is of unsupported type
    """
    filepath = Path(filename)
    if filepath.suffix.endswith("json"):
        with filepath.open() as fp:
            return json.load(fp)
    elif filepath.suffix.endswith(("yml", "yaml")):
        with filepath.open() as fp:
            return yaml.safe_load(fp)
    else:
        raise TypeError("config file must be json, or yaml/yml")

def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using existing file", filename, file=sys.stderr)
        return path

    if log:
        print("Downloading", url, file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)
