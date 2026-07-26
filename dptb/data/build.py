import os
import glob
from typing import Union
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data.transforms import OrbitalMapper
from dptb.utils.argcheck import collect_cutoffs
from dptb.utils.argcheck import get_cutoffs_from_model_options
import logging
import torch


log = logging.getLogger(__name__)


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

    def check_cutoffs(self,model:torch.nn.Module=None, **kwargs):
        if model is None:
            self.if_check_cutoffs = False
            log.warning("No model is provided. We can not check the cutoffs used in data and model are consistent.")
        else:
            self.if_check_cutoffs = True
            cutoff_options = collect_cutoffs(model.model_options)
            if isinstance(cutoff_options['r_max'],dict):
                assert isinstance(self.r_max,dict), "The r_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['r_max']:
                    if key not in self.r_max:
                        log.error(f"The key {key} in r_max is not defined in dataset")
                        raise ValueError(f"The key {key} in r_max is not defined in dataset")
                    assert self.r_max >=  cutoff_options['r_max'][key], f"The r_max in model shoule be  smaller than in dataset for {key}."

            elif isinstance(cutoff_options['r_max'],float):
                assert isinstance(self.r_max,float), "The r_max in model is a float, but in dataset it is not."
                assert self.r_max >=  cutoff_options['r_max'], "The r_max in model shoule be  smaller than in dataset."        

            if isinstance(cutoff_options['er_max'],dict):
                assert isinstance(self.er_max,dict), "The er_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['er_max']:
                    if key not in self.er_max:
                        log.error(f"The key {key} in er_max is not defined in dataset")
                        raise ValueError(f"The key {key} in er_max is not defined in dataset")

                    assert self.er_max >=  cutoff_options['er_max'][key], f"The er_max in model shoule be  smaller than in dataset for {key}."

            elif isinstance(cutoff_options['er_max'],float):
                assert isinstance(self.er_max,float), "The er_max in model is a float, but in dataset it is not."
                assert self.er_max >=  cutoff_options['er_max'], "The er_max in model shoule be  smaller than in dataset."
            elif cutoff_options['er_max'] is None:
                assert self.er_max is None, "The er_max in model is None, but in dataset it is not."


            if isinstance(cutoff_options['oer_max'],dict):
                assert isinstance(self.oer_max,dict), "The oer_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['oer_max']:
                    if key not in self.oer_max:
                        log.error(f"The key {key} in oer_max is not defined in dataset")
                        raise ValueError(f"The key {key} in oer_max is not defined in dataset")

                    assert self.oer_max >=  cutoff_options['oer_max'][key], f"The oer_max in model shoule be  smaller than in dataset for {key}."
            elif isinstance(cutoff_options['oer_max'],float):
                assert isinstance(self.oer_max,float), "The oer_max in model is a float, but in dataset it is not."
                assert self.oer_max >=  cutoff_options['oer_max'], "The oer_max in model shoule be  smaller than in dataset."
            elif cutoff_options['oer_max'] is None:
                assert self.oer_max is None, "The oer_max in model is None, but in dataset it is not."

build_dataset = DatasetBuilder()
