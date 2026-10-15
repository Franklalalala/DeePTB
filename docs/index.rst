=====================
DeePTB documentation
=====================

DeePTB provides Slater–Koster models and equivariant atomic-orbital operator prediction, including Hamiltonian, density, and overlap matrices, spin-orbit coupling, and electronic-structure postprocessing.

``1015-stable`` is the maintenance branch; its physical priors (``dptb.prior``) and inference (``dptb.inference``) run on the GPU. UniTB defaults to PDQ-MoE and supports a single-expert UniTB-dense configuration. Upstream ``lem`` / ``slem`` and prior-aware baselines have separate interfaces. SO2CUDA provides optional acceleration; LoopSCF uses DeePTB from its own repository.

.. toctree::
   :maxdepth: 2
   :caption: Architecture and data contracts

   architecture
   1015-stable
   1013-stable
   unitb
   embedding_baselines
   so2_backend
   advanced/so2_compact_wigner
   advanced/prior_inputs
   h0
   advanced/dynamic_batch_oom_fallback

.. toctree::
   :maxdepth: 2
   :caption: Quick start

   quick_start/easy_install
   quick_start/input
   quick_start/hands_on/index
   quick_start/basic_api

.. toctree::
   :maxdepth: 2
   :caption: Input fields

   input_params/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced usage

   advanced/sktb/index
   advanced/e3tb/index
   advanced/elec_properties/index
   advanced/interface/index

.. toctree::
   :maxdepth: 2
   :caption: Citations

   CITATIONS

.. toctree::
   :maxdepth: 2
   :caption: Development team

   DevelopingTeam

.. toctree::
   :maxdepth: 2
   :caption: Community

   community/contribution_guide
   CONTRIBUTING
