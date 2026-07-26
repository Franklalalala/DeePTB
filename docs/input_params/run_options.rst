Run Options
===========
run_op:
    | type: ``dict``
    | argument path: ``run_op``

    task_options:
        | type: ``dict``, optional
        | argument path: ``run_op/task_options``

        The maintained postprocessing task: band or write_block.


        Depending on the value of *task*, different sub args are accepted.

        task:
            | type: ``str`` (flag key)
            | argument path: ``run_op/task_options/task``
            | possible choices: band, write_block

            The string define the task DeePTB conduct, includes:
                                - `band`: for band structure plotting.
                                - `write_block`: write predicted Hamiltonian blocks.



        When *task* is set to ``band``:

        kline_type:
            | type: ``str``
            | argument path: ``run_op/task_options[band]/kline_type``

            The different type to build kpath line mode.
                                - "abacus" : the abacus format
                                - "vasp" : the vasp format
                                - "ase" : the ase format


        kpath:
            | type: ``list`` | ``str``
            | argument path: ``run_op/task_options[band]/kpath``

            for abacus, this is list of list of float, for vasp it is a list[str] to specify the kpath.

        high_sym_kpoints:
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[band]/high_sym_kpoints``

            the high symmetry kpoints dict, e.g. {'G':[0,0,0],'K':[0.5,0.5,0]}, only used for kline_type is vasp

        number_in_line:
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/number_in_line``

            the number of kpoints in each line path, only used for kline_type is vasp.

        klabels:
            | type: ``list``, optional, default: ``['']``
            | argument path: ``run_op/task_options[band]/klabels``

            the labels for high symmetry kpoint

        E_fermi:
            | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/E_fermi``

            the fermi level used to plot band

        emin:
            | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/emin``

            the min energy to show the band plot

        emax:
            | type: ``int`` | ``NoneType`` | ``float``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/emax``

            the max energy to show the band plot

        nkpoints:
            | type: ``int``, optional, default: ``0``
            | argument path: ``run_op/task_options[band]/nkpoints``

            the max energy to show the band plot

        ref_band:
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/ref_band``

            the reference band structure to be ploted together with dptb bands.

        nel_atom:
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/nel_atom``

            the valence electron number of each type of atom.

        override_overlap:
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/override_overlap``

            overlap file path to be input to override overlap matrix.

        eig_solver:
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/eig_solver``

            the eigenvalue solver to be used.


        When *task* is set to ``write_block``:


    structure:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/structure``

        the structure to run the task

    pbc:
        | type: ``list`` | ``bool`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/pbc``

        The periodic boundary condition, choose among:
                            Default: True,
                                - True: indicating the structure is periodic
                                - False: indicating the structure is not periodic
                                - list of bool: indicating the structure is periodic in x,y,z direction respectively.


    use_gui:
        | type: ``bool``, optional, default: ``False``
        | argument path: ``run_op/use_gui``

        To use the GUI or not

    device:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/device``

        The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: None. default None means to use the device seeting in the model ckpt file.

    dtype:
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/dtype``

        The digital number's precison, choose among:
                            Default: None,
                                - `float32`: indicating torch.float32
                                - `float64`: indicating torch.float64
                            default None means to use the device seeting in the model ckpt file.


    AtomicData_options:
        | type: ``dict`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/AtomicData_options``

        r_max:
            | type: ``int`` | ``dict`` | ``float``
            | argument path: ``run_op/AtomicData_options/r_max``

            the cutoff value for bond considering in TB model.

        er_max:
            | type: ``int`` | ``dict`` | ``NoneType`` | ``float``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/er_max``

            Optional environment-edge cutoff used by maintained embeddings with a distinct environment graph.

        oer_max:
            | type: ``int`` | ``dict`` | ``NoneType`` | ``float``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/oer_max``

            Optional onsite-environment cutoff used by maintained embeddings with a distinct onsite graph.
