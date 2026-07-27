import os
import glob
from typing import Union
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data.transforms import OrbitalMapper
from dptb.utils.argcheck import get_cutoffs_from_model_options
import logging
import torch


log = logging.getLogger(__name__)


def _validate_cutoff_coverage(name, dataset_cutoff, model_cutoff):
    """Require the dataset graph cutoff to cover the model cutoff."""

    numeric = (int, float)
    if model_cutoff is None:
        if dataset_cutoff is not None:
            raise ValueError(
                f"{name} is disabled in the model but set to {dataset_cutoff!r} "
                "for the dataset."
            )
        return

    if isinstance(model_cutoff, dict):
        if not isinstance(dataset_cutoff, dict):
            raise ValueError(
                f"The model {name} is a dict, but the dataset {name} is "
                f"{type(dataset_cutoff).__name__}."
            )
        missing = sorted(set(model_cutoff) - set(dataset_cutoff))
        if missing:
            raise ValueError(
                f"Dataset {name} is missing model cutoff keys: {missing}."
            )
        too_small = {
            key: (dataset_cutoff[key], required)
            for key, required in model_cutoff.items()
            if dataset_cutoff[key] < required
        }
        if too_small:
            raise ValueError(
                f"Dataset {name} must be greater than or equal to the model "
                f"cutoff for every key; offending values: {too_small}."
            )
        return

    if isinstance(model_cutoff, numeric) and not isinstance(model_cutoff, bool):
        if not isinstance(dataset_cutoff, numeric) or isinstance(dataset_cutoff, bool):
            raise ValueError(
                f"The model {name} is scalar, but the dataset {name} is "
                f"{type(dataset_cutoff).__name__}."
            )
        if dataset_cutoff < model_cutoff:
            raise ValueError(
                f"Dataset {name}={dataset_cutoff} is smaller than model "
                f"{name}={model_cutoff}."
            )
        return

    raise TypeError(f"Unsupported model {name} value: {model_cutoff!r}.")


class DatasetBuilder:
    def __init__(self):
        pass

    def __call__(
        self,
        root: str,
        r_max: Union[float, int, dict],
        er_max: Union[float, int, dict] = None,
        oer_max: Union[float, int, dict] = None,
        type: str = "LMDBDataset",
        prefix: str = None,
        separator: str = ".",
        get_Hamiltonian: bool = False,
        get_overlap: bool = False,
        get_DM: bool = False,
        get_eigenvalues: bool = False,
        orthogonal: bool = False,
        basis: str = None,
        **kwargs,
    ):
        """Build a maintained LMDB-backed training or evaluation dataset."""
        if type != "LMDBDataset":
            raise ValueError(
                "0726-light supports only data_options.*.type='LMDBDataset'; "
                f"got {type!r}."
            )
        if prefix is None:
            raise ValueError(
                "prefix is required to select LMDB shard directories."
            )

        self.r_max = r_max
        self.er_max = er_max
        self.oer_max = oer_max
        self.if_check_cutoffs = False

        if basis is not None:
            idp = OrbitalMapper(
                basis=basis,
                has_soc=kwargs.get("has_soc", False),
                soc_complex_doubling=kwargs.get(
                    "soc_complex_doubling", True
                ),
                nextham_uureal_mask=kwargs.get(
                    "nextham_uureal_mask", False
                ),
                full_soc_prediction=kwargs.get(
                    "full_soc_prediction", False
                ),
            )
        else:
            idp = None

        pattern = os.path.join(root, f"{prefix}{separator}*")
        include_folders = []
        for directory in glob.glob(pattern):
            if not os.path.isdir(directory):
                continue
            if not glob.glob(os.path.join(directory, "*.mdb")):
                raise ValueError(
                    f"{directory} does not contain an LMDB .mdb file."
                )
            include_folders.append(
                os.path.basename(os.path.normpath(directory))
            )
        if not include_folders:
            raise ValueError(
                f"No LMDB shard directories match {pattern!r}."
            )

        shared_info = {
            "r_max": r_max,
            "er_max": er_max,
            "oer_max": oer_max,
            "wave_align": kwargs.get("wave_align", False),
            "train_w_homo_lumo_gap": kwargs.get(
                "train_w_homo_lumo_gap", False
            ),
            "train_w_eps": kwargs.get("train_w_eps", False),
            "train_w_charge": kwargs.get("train_w_charge", False),
            "train_dip": kwargs.get("train_dip", False),
            "train_polar": kwargs.get("train_polar", False),
        }
        info_files = {
            name: dict(shared_info) for name in sorted(include_folders)
        }

        dataset = LMDBDataset(
            root=root,
            type_mapper=idp,
            orthogonal=orthogonal,
            get_Hamiltonian=get_Hamiltonian,
            get_H0=kwargs.get("get_H0", False),
            get_prior=kwargs.get(
                "get_prior", kwargs.get("get_P2", False)
            ),
            residual_hamiltonian=kwargs.get(
                "residual_hamiltonian", False
            ),
            residual_shrink_policy=kwargs.get(
                "residual_shrink_policy", "error"
            ),
            min_residual_shrink=kwargs.get(
                "min_residual_shrink", 1.2
            ),
            get_overlap=get_overlap,
            get_DM=get_DM,
            get_eigenvalues=get_eigenvalues,
            h0_key=kwargs.get("h0_key", "hamiltonian_0"),
            prefer_precomputed_h0=kwargs.get(
                "prefer_precomputed_h0", True
            ),
            prior_kind=kwargs.get("prior_kind", "p2"),
            prior_raw_key=kwargs.get(
                "prior_raw_key", kwargs.get("p2_key", None)
            ),
            prefer_precomputed_prior=kwargs.get(
                "prefer_precomputed_prior",
                kwargs.get("prefer_precomputed_p2", True),
            ),
            require_prior_blocks=kwargs.get(
                "require_prior_blocks",
                kwargs.get("require_p2_blocks", False),
            ),
            require_full_h_target=kwargs.get(
                "require_full_h_target", False
            ),
            require_residual_h_target=kwargs.get(
                "require_residual_h_target", False
            ),
            require_uureal_block_ode=kwargs.get(
                "require_uureal_block_ode", False
            ),
            require_residual_from_full_h_target=kwargs.get(
                "require_residual_from_full_h_target", False
            ),
            expected_prior_source_fingerprint=kwargs.get(
                "expected_prior_source_fingerprint",
                kwargs.get("expected_p2_source_fingerprint", ""),
            ),
            expected_physical_h0_source_fingerprint=kwargs.get(
                "expected_physical_h0_source_fingerprint", ""
            ),
            audit_prior_representations=kwargs.get(
                "audit_prior_representations",
                kwargs.get("audit_p2_representations", False),
            ),
            info_files=info_files,
        )

        log.warning(
            "The cutoffs in data and model are not checked. Be careful!"
        )
        return dataset

    def from_model(self,
               model, 
               root: str,
               type: str = "LMDBDataset",
               prefix: str = None,
               separator:str='.',
               get_Hamiltonian: bool = False,
               get_overlap: bool = False,
               get_DM: bool = False,
               get_eigenvalues: bool = False,
               # common_options
               orthogonal: bool = False,
               basis: str = None, 
               **kwargs):
        """
        Build a dataset from a model.

        Args:
            - model (torch.nn.Module): The model to build the dataset from.
            - dataset_type (str, optional): Maintained type; only LMDBDataset.

        Returns:
            dataset: The built dataset.
        """
        # cutoff_options = collect_cutoffs(model.model_options)
        r_max, er_max, oer_max  = get_cutoffs_from_model_options(model.model_options)
        cutoff_options = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}

        dataset = self(
            root = root,
            **cutoff_options,
            type = type,
            prefix = prefix,
            separator = separator,
            get_Hamiltonian = get_Hamiltonian,
            get_overlap = get_overlap,
            get_DM = get_DM,
            get_eigenvalues = get_eigenvalues,
            orthogonal = orthogonal,
            basis = basis, 
            **kwargs,
        )

        return dataset

    def check_cutoffs(self, model: torch.nn.Module = None, **kwargs):
        if model is None:
            self.if_check_cutoffs = False
            log.warning("No model is provided. We can not check the cutoffs used in data and model are consistent.")
            return

        # collect_cutoffs() indexes jdata["model_options"]; passing
        # model.model_options directly raised KeyError: 'model_options' on the
        # first statement, which is why this guard never ran.
        r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
        model_cutoffs = {"r_max": r_max, "er_max": er_max, "oer_max": oer_max}
        for name, model_cutoff in model_cutoffs.items():
            _validate_cutoff_coverage(name, getattr(self, name, None), model_cutoff)
        self.if_check_cutoffs = True

build_dataset = DatasetBuilder()
