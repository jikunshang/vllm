MODEL=Intel/gpt-oss-20b-int4-AutoRound
TP=2
PP=4

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1

# export VLLM_TORCH_PROFILER_DIR=${PWD}/profile
export TORCH_LLM_ALLREDUCE=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

# vllm serve ${MODEL} --async-scheduling --enforce-eager
python examples/offline_inference/basic/generate.py \
  --distributed-executor-backend mp \
  --model ${MODEL} \
  --model_impl transformers \
  --max-model-len 1024 \
  --max-num-batched-tokens 2048 \
  --enforce-eager
