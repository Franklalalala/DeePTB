=================
DeePTB 文档
=================

DeePTB 提供 Slater–Koster 模型与等变原子轨道算符预测，支持哈密顿量、密度矩阵、重叠矩阵、自旋轨道耦合以及电子结构后处理。

``1006-stable`` 是本仓库的维护分支。UniTB 默认使用 PDQ-MoE，并支持单专家的 UniTB-dense 配置；上游无先验 ``lem`` / ``slem`` 与带先验基线独立维护。SO2CUDA 是可选加速组件，LoopSCF 在独立仓库中使用 DeePTB 的通用接口。

.. toctree::
   :maxdepth: 2
   :caption: 维护版本与数据契约

   1006-stable
   advanced/prior_inputs
   nacf_candidate_prior
   nacf_compact_packing
   nacf_native_assembly
   nacf_edge_vna
   nacf_prepared_store
   advanced/dynamic_batch_oom_fallback

.. toctree::
   :maxdepth: 2
   :caption: 快速开始
   

   quick_start/easy_install
   quick_start/input
   quick_start/hands_on/index
   quick_start/basic_api


.. toctree::
   :maxdepth: 2
   :caption: 输入字段
   
   input_params/index


.. toctree::
   :maxdepth: 2
   :caption: 进阶用法
   
   advanced/sktb/index
   advanced/e3tb/index
   advanced/elec_properties/index
   advanced/interface/index

.. toctree::
   :maxdepth: 2
   :caption: 引用

   CITATIONS

.. toctree::
   :maxdepth: 2
   :caption: 开发团队

   DevelopingTeam

.. toctree::
   :maxdepth: 2
   :caption: 社区

   community/contribution_guide
   CONTRIBUTING

