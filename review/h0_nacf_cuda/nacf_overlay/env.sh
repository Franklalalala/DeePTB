export NACF_CUDA_ROOT=/home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate/nacf_candidate
source /home/mingkang_nt/codex/h0_cuda_production_gemini_20260912/review_candidate/env.sh
export PYTHONPATH="$NACF_CUDA_ROOT:$PYTHONPATH"
export DPTB_NACF_PREPARED_DIR="$NACF_CUDA_ROOT/prepared_tables"
mkdir -p "$DPTB_NACF_PREPARED_DIR"
