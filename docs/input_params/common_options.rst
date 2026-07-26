Common Options
==============
common_options:
    | type: ``dict``
    | argument path: ``common_options``

    basis:
        | type: ``dict``
        | argument path: ``common_options/basis``

        The atomic orbitals used to construct the basis. e.p. {'A':['2s','2p','s*'],'B':'[3s','3p']}

    overlap:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/overlap``

        Whether to calculate the overlap matrix. Default: False

    train_polar:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/train_polar``

        Whether to train the polarizaty tensor. Default: False

    wave_align:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/wave_align``

        Whether to align the wavefunctions. Default: False

    train_dip:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/train_dip``

        Whether to train the dipole moment tensor. Default: False

    train_w_charge:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/train_w_charge``

        Whether to train with charge info. Default: False

    has_soc:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/has_soc``

        Whether to train with SOC. Default: False

    nextham_uureal_mask:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/nextham_uureal_mask``

        Whether to expose the NextHAM SOC uu.real mask to dataset and loss helpers. Default: False

    full_soc_prediction:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/full_soc_prediction``

        Whether to predict the full SOC target space. When True, this overrides nextham_uureal_mask and restores all spin and real/imag SOC channels. Default: False

    device:
        | type: ``str``, optional, default: ``cpu``
        | argument path: ``common_options/device``

        The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`

    dtype:
        | type: ``str``, optional, default: ``float32``
        | argument path: ``common_options/dtype``

        The digital number's precison, choose among:
                            Default: `float32`
                                - `float32`: indicating torch.float32
                                - `float64`: indicating torch.float64


    seed:
        | type: ``int``, optional, default: ``3982377700``
        | argument path: ``common_options/seed``

        The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`
