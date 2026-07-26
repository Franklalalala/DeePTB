"""Shared constants used by the maintained E3/LMDB model path."""

import ase
import numpy as np
import torch
from scipy.constants import Boltzmann


CUBIC_MAG_NUM_DICT = {
    "s": [0],
    "p": [-1, 0, 1],
    "d": [-2, -1, 0, 1, 2],
}
LM_MAG_NUM_DICT = {
    "s": [0],
    "p": [-1, 0, 1],
    "d": [-2, -1, 0, 1, 2],
}

anglrMId = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
orbitalId = {value: key for key, value in anglrMId.items()}

Bohr2Ang = 0.529177210903
eV2J = 1.6021766208e-19
dtype_dict = {"float32": torch.float32, "float64": torch.float64}

atomic_num_dict = ase.atom.atomic_numbers
atomic_num_dict_r = dict(
    zip(atomic_num_dict.values(), atomic_num_dict.keys())
)

ABACUS2DeePTB = {
    0: np.eye(1, dtype=np.float32),
    1: np.eye(3, dtype=np.float32)[[2, 0, 1]],
    2: np.eye(5, dtype=np.float32)[[4, 2, 0, 1, 3]],
    3: np.eye(7, dtype=np.float32)[[6, 4, 2, 0, 1, 3, 5]],
    4: np.eye(9, dtype=np.float32)[[8, 6, 4, 2, 0, 1, 3, 5, 7]],
    5: np.eye(11, dtype=np.float32)[
        [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]
    ],
}
ABACUS2DeePTB[1][[0, 2]] *= -1
ABACUS2DeePTB[2][[1, 3]] *= -1
ABACUS2DeePTB[3][[0, 6, 2, 4]] *= -1
ABACUS2DeePTB[4][[1, 7, 3, 5]] *= -1
ABACUS2DeePTB[5][[0, 10, 8, 2, 6, 4]] *= -1

OPENMX2DeePTB = {
    "s": torch.eye(1).double(),
    "p": torch.eye(3)[[1, 2, 0]].double(),
    "d": torch.eye(5)[[2, 4, 0, 3, 1]].double(),
    "f": torch.eye(7)[[6, 4, 2, 0, 1, 3, 5]].double(),
}
