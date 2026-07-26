Data Options
============
data_options:
    | type: ``dict``
    | argument path: ``data_options``

    The options for dataset settings in training.

    r_max:
        | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
        | argument path: ``data_options/r_max``

        r_max

    oer_max:
        | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
        | argument path: ``data_options/oer_max``

        oer_max

    er_max:
        | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
        | argument path: ``data_options/er_max``

        er_max

    train:
        | type: ``dict``
        | argument path: ``data_options/train``

        LMDB dataset settings for training.

        type:
            | type: ``str``, optional, default: ``LMDBDataset``
            | argument path: ``data_options/train/type``

            The maintained dataset backend. Only LMDBDataset is supported.

        root:
            | type: ``str``
            | argument path: ``data_options/train/root``

            Root containing LMDB shard directories.

        prefix:
            | type: ``str``
            | argument path: ``data_options/train/prefix``

            Shard-directory prefix.

        separator:
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/train/separator``

            Prefix/suffix separator.

        get_Hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_Hamiltonian``

            Load Hamiltonian blocks.

        get_H0:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_H0``

            Load physical H0 initialization data.

        get_P2:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_P2``

            Backward-compatible switch for the selected P2/P23 prior.

        prior_kind:
            | type: ``str``, optional, default: ``p2``
            | argument path: ``data_options/train/prior_kind``

            Selected physical prior: p2 or p23.

        residual_hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/residual_hamiltonian``

            Train on dH = H - H0.

        residual_shrink_policy:
            | type: ``str``, optional, default: ``error``
            | argument path: ``data_options/train/residual_shrink_policy``

            Residual shrink gate: error, warn, or off.

        min_residual_shrink:
            | type: ``int`` | ``float``, optional, default: ``1.2``
            | argument path: ``data_options/train/min_residual_shrink``

            Minimum residual shrink ratio.

        h0_key:
            | type: ``str``, optional, default: ``hamiltonian_0``
            | argument path: ``data_options/train/h0_key``

            Raw LMDB H0 key.

        prefer_precomputed_h0:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/train/prefer_precomputed_h0``

            Prefer stored node_h0/edge_h0 features.

        p2_key:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/train/p2_key``

            Deprecated explicit raw prior key; normally derived from prior_kind.

        prefer_precomputed_p2:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/train/prefer_precomputed_p2``

            Prefer stored selected-prior RME features.

        require_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/require_full_h_target``

            Require versioned absolute Full-H target provenance.

        require_residual_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/require_residual_h_target``

            Require versioned residual-H target provenance.

        require_uureal_block_ode:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/require_uureal_block_ode``

            Require compact uu_real block-ODE records.

        require_residual_from_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/require_residual_from_full_h_target``

            Require online H-H0 materialization from absolute Full H.

        expected_p2_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/train/expected_p2_source_fingerprint``

            Expected selected-prior source SHA256.

        expected_physical_h0_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/train/expected_physical_h0_source_fingerprint``

            Expected physical-H0 source SHA256.

        allow_unbound_prior_source_fingerprint:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/allow_unbound_prior_source_fingerprint``

            Development-only prior provenance escape hatch.

        audit_p2_representations:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/audit_p2_representations``

            Audit selected-prior RME/AO consistency at ingest.

        require_p2_blocks:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/require_p2_blocks``

            Require selected-prior AO blocks.

        get_overlap:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_overlap``

            Load overlap blocks.

        get_DM:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_DM``

            Load density matrices.

        get_eigenvalues:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_eigenvalues``

            Load eigenvalues and k-points.

    validation:
        | type: ``dict``, optional
        | argument path: ``data_options/validation``

        LMDB dataset settings for validation.

        type:
            | type: ``str``, optional, default: ``LMDBDataset``
            | argument path: ``data_options/validation/type``

            The maintained dataset backend. Only LMDBDataset is supported.

        root:
            | type: ``str``
            | argument path: ``data_options/validation/root``

            Root containing LMDB shard directories.

        prefix:
            | type: ``str``
            | argument path: ``data_options/validation/prefix``

            Shard-directory prefix.

        separator:
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/validation/separator``

            Prefix/suffix separator.

        get_Hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_Hamiltonian``

            Load Hamiltonian blocks.

        get_H0:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_H0``

            Load physical H0 initialization data.

        get_P2:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_P2``

            Backward-compatible switch for the selected P2/P23 prior.

        prior_kind:
            | type: ``str``, optional, default: ``p2``
            | argument path: ``data_options/validation/prior_kind``

            Selected physical prior: p2 or p23.

        residual_hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/residual_hamiltonian``

            Train on dH = H - H0.

        residual_shrink_policy:
            | type: ``str``, optional, default: ``error``
            | argument path: ``data_options/validation/residual_shrink_policy``

            Residual shrink gate: error, warn, or off.

        min_residual_shrink:
            | type: ``int`` | ``float``, optional, default: ``1.2``
            | argument path: ``data_options/validation/min_residual_shrink``

            Minimum residual shrink ratio.

        h0_key:
            | type: ``str``, optional, default: ``hamiltonian_0``
            | argument path: ``data_options/validation/h0_key``

            Raw LMDB H0 key.

        prefer_precomputed_h0:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/validation/prefer_precomputed_h0``

            Prefer stored node_h0/edge_h0 features.

        p2_key:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/validation/p2_key``

            Deprecated explicit raw prior key; normally derived from prior_kind.

        prefer_precomputed_p2:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/validation/prefer_precomputed_p2``

            Prefer stored selected-prior RME features.

        require_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/require_full_h_target``

            Require versioned absolute Full-H target provenance.

        require_residual_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/require_residual_h_target``

            Require versioned residual-H target provenance.

        require_uureal_block_ode:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/require_uureal_block_ode``

            Require compact uu_real block-ODE records.

        require_residual_from_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/require_residual_from_full_h_target``

            Require online H-H0 materialization from absolute Full H.

        expected_p2_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/validation/expected_p2_source_fingerprint``

            Expected selected-prior source SHA256.

        expected_physical_h0_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/validation/expected_physical_h0_source_fingerprint``

            Expected physical-H0 source SHA256.

        allow_unbound_prior_source_fingerprint:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/allow_unbound_prior_source_fingerprint``

            Development-only prior provenance escape hatch.

        audit_p2_representations:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/audit_p2_representations``

            Audit selected-prior RME/AO consistency at ingest.

        require_p2_blocks:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/require_p2_blocks``

            Require selected-prior AO blocks.

        get_overlap:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_overlap``

            Load overlap blocks.

        get_DM:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_DM``

            Load density matrices.

        get_eigenvalues:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_eigenvalues``

            Load eigenvalues and k-points.

    reference:
        | type: ``dict``, optional
        | argument path: ``data_options/reference``

        LMDB dataset settings for reference batches.

        type:
            | type: ``str``, optional, default: ``LMDBDataset``
            | argument path: ``data_options/reference/type``

            The maintained dataset backend. Only LMDBDataset is supported.

        root:
            | type: ``str``
            | argument path: ``data_options/reference/root``

            Root containing LMDB shard directories.

        prefix:
            | type: ``str``
            | argument path: ``data_options/reference/prefix``

            Shard-directory prefix.

        separator:
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/reference/separator``

            Prefix/suffix separator.

        get_Hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_Hamiltonian``

            Load Hamiltonian blocks.

        get_H0:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_H0``

            Load physical H0 initialization data.

        get_P2:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_P2``

            Backward-compatible switch for the selected P2/P23 prior.

        prior_kind:
            | type: ``str``, optional, default: ``p2``
            | argument path: ``data_options/reference/prior_kind``

            Selected physical prior: p2 or p23.

        residual_hamiltonian:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/residual_hamiltonian``

            Train on dH = H - H0.

        residual_shrink_policy:
            | type: ``str``, optional, default: ``error``
            | argument path: ``data_options/reference/residual_shrink_policy``

            Residual shrink gate: error, warn, or off.

        min_residual_shrink:
            | type: ``int`` | ``float``, optional, default: ``1.2``
            | argument path: ``data_options/reference/min_residual_shrink``

            Minimum residual shrink ratio.

        h0_key:
            | type: ``str``, optional, default: ``hamiltonian_0``
            | argument path: ``data_options/reference/h0_key``

            Raw LMDB H0 key.

        prefer_precomputed_h0:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/reference/prefer_precomputed_h0``

            Prefer stored node_h0/edge_h0 features.

        p2_key:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/reference/p2_key``

            Deprecated explicit raw prior key; normally derived from prior_kind.

        prefer_precomputed_p2:
            | type: ``bool``, optional, default: ``True``
            | argument path: ``data_options/reference/prefer_precomputed_p2``

            Prefer stored selected-prior RME features.

        require_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/require_full_h_target``

            Require versioned absolute Full-H target provenance.

        require_residual_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/require_residual_h_target``

            Require versioned residual-H target provenance.

        require_uureal_block_ode:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/require_uureal_block_ode``

            Require compact uu_real block-ODE records.

        require_residual_from_full_h_target:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/require_residual_from_full_h_target``

            Require online H-H0 materialization from absolute Full H.

        expected_p2_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/reference/expected_p2_source_fingerprint``

            Expected selected-prior source SHA256.

        expected_physical_h0_source_fingerprint:
            | type: ``str``, optional, default: (empty string)
            | argument path: ``data_options/reference/expected_physical_h0_source_fingerprint``

            Expected physical-H0 source SHA256.

        allow_unbound_prior_source_fingerprint:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/allow_unbound_prior_source_fingerprint``

            Development-only prior provenance escape hatch.

        audit_p2_representations:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/audit_p2_representations``

            Audit selected-prior RME/AO consistency at ingest.

        require_p2_blocks:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/require_p2_blocks``

            Require selected-prior AO blocks.

        get_overlap:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_overlap``

            Load overlap blocks.

        get_DM:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_DM``

            Load density matrices.

        get_eigenvalues:
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_eigenvalues``

            Load eigenvalues and k-points.
