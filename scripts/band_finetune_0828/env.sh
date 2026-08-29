# Environment for the band fine-tune on gq6000 (RTX PRO 6000 Blackwell, sm120).
#
# The model runs so2_fusion_mode=streamed_m_major_fused_p0 and
# mole_linear_mode=cublas_grouped. Both import so2_cuda_ops, which is not an
# installed package but a source tree that must be on PYTHONPATH, and whose
# CUDA kernels are JIT-built on first call.
#
# CUDA_HOME is the part that is easy to get wrong: /usr/local/cuda is 12.4,
# whose nvcc rejects compute_120 outright ("Unsupported gpu architecture").
# torch itself is cu128, so imports and non-kernel paths look healthy and the
# failure only surfaces at the first kernel build, several minutes into a run.
# The 12.8 toolkit below is the one the cached sm120 objects were built with.
export CUDA_HOME=/home/user/codex/cuda-12.8-sm120_20260525
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0+PTX

export PYTHONPATH=/data/wgh/p2p23_band_0804/code/SO2CUDA/src
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=8

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate moe0309
