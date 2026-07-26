import torch.nn as nn
import torch
from typing import Union, Optional, Dict
from dptb.configuration import (
    canonicalize_embedding_options,
    canonicalize_prediction_options,
    migrate_legacy_checkpoint_model_options,
    resolve_reconstruction_mode,
)
from dptb.nn.embedding import Embedding
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
from dptb.data.interfaces.p2_contract import (
    build_prior_spec,
    validate_explicit_prior_fields,
)
from dptb.nn.blockwise_hamiltonian import (
    BlockwiseE3Hamiltonian,
    attach_full_hamiltonian_from_h0,
    attach_full_hamiltonian_from_prior,
)
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nn.rescale import E3PerSpeciesScaleShift, E3PerEdgeSpeciesScaleShift
from dptb.nn.embedding.output_routes import resolve_output_route, validate_prediction_route
from dptb.utils.soc_target import resolve_nextham_uureal_mask
import logging

log = logging.getLogger(__name__)


def _resolve_embedding_output_route_spec(embedding_module, embedding_options, prediction_options):
    spec = getattr(embedding_module, "output_route_spec", None)
    if spec is not None:
        return spec

    spec = resolve_output_route(
        output_route=embedding_options.get("output_route", prediction_options.get("output_route", None)),
        legacy_mode=embedding_options.get("rme_head_mode", prediction_options.get("rme_head_mode", None)),
        projector_backend=embedding_options.get(
            "ao_projector_backend",
            prediction_options.get("ao_projector_backend", None),
        ),
        projector_bank_path=embedding_options.get(
            "ao_projector_bank_path",
            prediction_options.get("ao_projector_bank_path", None),
        ),
    )
    if spec.canonical_name != "legacy_rme":
        raise RuntimeError(
            "Embedding did not expose output_route_spec for configured "
            f"output route {spec.canonical_name!r}; route validation cannot be performed."
        )
    log.info("Embedding did not expose output_route_spec; assuming legacy_rme route.")
    return spec

class NNENV(nn.Module):
    quantities = ["hamiltonian", "energy"]
    name = "nnenv"
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            overlap: bool = False,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            has_soc: bool = False,
            scale_type: str = 'scale_w_back_grad',
            **kwargs,
    ):
        
        """The top level DeePTB model class.

        Parameters
        ----------
        embedding_config : dict
            _description_
        prediction_config : dict
            _description_
        basis : Dict[str, Union[str, list], None], optional
            _description_, by default None
        idp : Union[OrbitalMapper, None], optional
            _description_, by default None
        transform : bool, optional
            _description_, decide whether to transform the irreducible matrix element to the hamiltonians
        dtype : Union[str, torch.dtype], optional
            _description_, by default torch.float32
        device : Union[str, torch.device], optional
            _description_, by default torch.device("cpu")

        Raises
        ------
        NotImplementedError
            _description_
        """
        super(NNENV, self).__init__()

        embedding = canonicalize_embedding_options(embedding)
        prediction = canonicalize_prediction_options(prediction)

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.model_options = {"embedding": embedding.copy(), "prediction": prediction.copy()}
        self.transform = transform

        self.method = prediction.get("method", "e3tb")
        if self.method not in {"e3tb", "block_native"}:
            raise ValueError(
                "0726-light supports prediction.method='e3tb' or 'block_native'."
            )
        if overlap:
            raise ValueError(
                "0726-light removed the legacy SK overlap prediction route."
            )
        # self.soc = prediction.get("soc", False)
        self.prediction = prediction

        prediction_copy = prediction.copy()
        scale_type = prediction_copy.get("scale_type")
        self.scale_type = scale_type
        self.blockwise_hamiltonian = bool(prediction_copy.get("blockwise_hamiltonian", False))
        self.reconstruction = resolve_reconstruction_mode(
            prediction_copy.get("reconstruction", "direct")
        )
        self.block_native_add_h0 = self.reconstruction == "h0_residual"
        self.block_native_add_prior = self.reconstruction == "prior_residual"
        # The prior AO-block fields and label are DERIVED from the single prior
        # kind carried by the embedding.  Explicit prediction fields are honored
        # only as overrides; when omitted (or left empty) they derive from the
        # kind so the RME conditioning and the AO reconstruction always agree.
        embedding_prior_kind = str(
            embedding.get("prior_kind", "p2") or "p2"
        ).strip().lower()
        try:
            block_native_prior_spec = build_prior_spec(embedding_prior_kind)
        except ValueError:
            block_native_prior_spec = build_prior_spec("p2")
        # Fail closed on direct-API construction: any explicit (non-empty)
        # prior field passed through the embedding/prediction dicts must agree
        # with the spec derived from the single prior kind, so p2 RME
        # conditioning can never be mixed with p23 AO blocks (or vice versa).
        validate_explicit_prior_fields(
            block_native_prior_spec,
            prior_node_key=embedding.get("prior_node_key"),
            prior_edge_key=embedding.get("prior_edge_key"),
            prior_node_block_field=prediction_copy.get("prior_node_block_field"),
            prior_edge_block_field=prediction_copy.get("prior_edge_block_field"),
            prior_label=prediction_copy.get("prior_label"),
        )
        self.block_native_prior_kind = block_native_prior_spec.kind
        self.block_native_prior_node_field = (
            prediction_copy.get("prior_node_block_field")
            or block_native_prior_spec.node_blocks_key
        )
        self.block_native_prior_edge_field = (
            prediction_copy.get("prior_edge_block_field")
            or block_native_prior_spec.edge_blocks_key
        )
        self.block_native_prior_label = (
            prediction_copy.get("prior_label")
            or block_native_prior_spec.label
        )
        self.block_native_validate_prior_blocks = bool(
            prediction_copy.get("validate_prior_blocks", False)
        )
        self.block_native_full_output_node_field = prediction_copy.get(
            "full_output_node_field", "node_full_hamil_blocks"
        )
        self.block_native_full_output_edge_field = prediction_copy.get(
            "full_output_edge_field", "edge_full_hamil_blocks"
        )

        self.has_soc = has_soc
        self.full_soc_prediction = bool(
            kwargs.get(
                "full_soc_prediction",
                embedding.get("full_soc_prediction", prediction.get("full_soc_prediction", False)),
            )
        )
        self.nextham_uureal_mask = resolve_nextham_uureal_mask(
            nextham_uureal_mask=kwargs.get(
                "nextham_uureal_mask",
                embedding.get("nextham_uureal_mask", prediction.get("nextham_uureal_mask", False)),
            ),
            full_soc_prediction=self.full_soc_prediction,
        )
        print(f'NNENV soc flag: {self.has_soc}')

        mapper_method = "e3tb" if self.method == "block_native" else self.method
        if basis is not None:
            self.idp = OrbitalMapper(
                basis,
                method=mapper_method,
                device=self.device,
                has_soc=has_soc,
                nextham_uureal_mask=self.nextham_uureal_mask,
                full_soc_prediction=self.full_soc_prediction,
            )
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.nextham_uureal_mask = bool(getattr(self.idp, "nextham_uureal_mask", self.nextham_uureal_mask))
        self.idp.get_orbpair_maps()

        embedding.update({
            'has_soc': has_soc,
            'nextham_uureal_mask': self.nextham_uureal_mask,
            'full_soc_prediction': self.full_soc_prediction,
        })

        n_species = len(self.basis.keys())
        # initialize the embedding layer
        self.embedding = Embedding(**embedding, dtype=dtype, device=device, idp=self.idp, n_atom=n_species)
        self.output_route_spec = _resolve_embedding_output_route_spec(
            self.embedding,
            embedding,
            prediction_copy,
        )
        validate_prediction_route(self.output_route_spec, prediction_copy)
        if self.output_route_spec.debug_only:
            log.warning(
                "Using debug-only non-equivariant output route %s.",
                self.output_route_spec.canonical_name,
            )
        
        # initialize the maintained prediction layer
        if prediction_copy.get("method") == "e3tb":
            self.node_prediction_h = E3PerSpeciesScaleShift(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                num_types=n_species,
                irreps_in=self.embedding.out_node_irreps,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                shifts=0.0,
                scales=1.0,
                dtype=self.dtype,
                device=self.device,
                **prediction_copy,
            )
            self.edge_prediction_h = E3PerEdgeSpeciesScaleShift(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                num_types=n_species,
                irreps_in=self.embedding.out_edge_irreps,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                shifts=0.0,
                scales=1.0,
                dtype=self.dtype,
                device=self.device,
                **prediction_copy,
            )
        elif prediction_copy.get("method") == "block_native":
            if overlap:
                raise NotImplementedError("block_native prediction does not support overlap.")
            if self.scale_type not in (None, "no_scale"):
                raise ValueError("block_native prediction requires scale_type='no_scale' or omitted.")
        else:
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction_copy["method"]))

        if self.method == "e3tb":
            hamiltonian_cls = BlockwiseE3Hamiltonian if self.blockwise_hamiltonian else E3Hamiltonian
            blockwise_ham_kwargs = {}
            if self.blockwise_hamiltonian:
                blockwise_ham_kwargs.update(
                    add_h0=self.block_native_add_h0,
                    add_prior=self.block_native_add_prior,
                    prior_kind=self.block_native_prior_kind,
                )
                for key in (
                    "node_pad_shape",
                    "edge_pad_shape",
                    "symmetrize_onsite",
                    "complete_edges",
                    "strict_complete_edges",
                    "prior_node_block_field",
                    "prior_edge_block_field",
                    "prior_label",
                    "validate_prior_blocks",
                    "full_output_node_field",
                    "full_output_edge_field",
                ):
                    if key in prediction_copy:
                        blockwise_ham_kwargs[key] = prediction_copy[key]

            self.hamiltonian = hamiltonian_cls(
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                idp=self.embedding.idp,
                dtype=self.dtype,
                device=self.device,
                soc=self.has_soc,
                nextham_uureal_mask=self.nextham_uureal_mask,
                **blockwise_ham_kwargs,
            )
        elif self.method == "block_native":
            pass


    def forward(self, data: AtomicDataDict.Type):
        if data.get(AtomicDataDict.EDGE_TYPE_KEY, None) is None:
            self.idp(data)

        data = self.embedding(data)
        if self.method == "block_native":
            if self.block_native_add_h0:
                attach_full_hamiltonian_from_h0(
                    data,
                    full_output_node_field=self.block_native_full_output_node_field,
                    full_output_edge_field=self.block_native_full_output_edge_field,
                    validate_prior_blocks=self.block_native_validate_prior_blocks,
                )
            elif self.block_native_add_prior:
                attach_full_hamiltonian_from_prior(
                    data,
                    prior_node_field=self.block_native_prior_node_field,
                    prior_edge_field=self.block_native_prior_edge_field,
                    full_output_node_field=self.block_native_full_output_node_field,
                    full_output_edge_field=self.block_native_full_output_edge_field,
                    prior_label=self.block_native_prior_label,
                    non_soc_only=True,
                    validate_prior_blocks=self.block_native_validate_prior_blocks,
                )
            return data
        if self.scale_type != "no_scale":
            data = self.node_prediction_h(data)
            data = self.edge_prediction_h(data)

        if self.transform:
            data = self.hamiltonian(data)

        return data
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint, 
        embedding: Optional[dict]=None,
        prediction: Optional[dict]=None,
        overlap: bool=None,
        basis: Dict[str, Union[str, list]]=None,
        dtype: Union[str, torch.dtype]=None,
        device: Union[str, torch.device]=None,
        transform: bool = True,
        **kwargs
        ):
        if device == 'cuda':
            if not torch.cuda.is_available():
                device = 'cpu'
                log.warning("CUDA is not available. The model will be loaded on CPU.")

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        checkpoint_model_options = migrate_legacy_checkpoint_model_options(
            ckpt["config"]["model_options"]
        )
        model_options = {
            "embedding": checkpoint_model_options["embedding"] if not embedding else embedding,
            "prediction": checkpoint_model_options["prediction"] if not prediction else prediction,
        }
        common_options = dict(ckpt["config"]["common_options"])
        common_options.update(kwargs)
        for key, value in {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }.items():
            if value is not None:
                common_options[key] = value
        model = cls(**model_options, **common_options, transform=transform)
        model.load_state_dict(ckpt["model_state_dict"])

        del ckpt

        return model
