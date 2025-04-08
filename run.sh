#!/bin/bash

: ${MODEL=${1:-"deepseek-ai/DeepSeek-R1"}} # HF model id
: ${TP=${2:-8}}                            # tensor-parallel
: ${EP=${4:-1}}                            # expert-parallel - (MOE)
: ${PP=${3:-1}}                            # pipeline-parallel - (AsyncLLMEngine)-WIP
: ${DP=${5:-1}}                            # data-parallel - (MLA)-WIP
: ${DUMMY=${6:--1}}                        # number dummy layers - default -1 to disable dummy weight
: ${IN=${7:-1024}}                         # input prompt length
: ${OUT=${8:-4}}                           # output generate length
: ${BS=${9:-1}}                            # batch size
: ${BEAM=${10:-1}}                         # beam width
: ${PROFILE=${11:-1}}                      # enable profile
: ${XPU_CCL_BACKEND=${12:-"xccl"}}         # ccl, xccl
: ${BENCHMARK=${13:-0}}                    # benchmark LATENCY | THROUGHPUT | SERVER

export VLLM_USE_V1=1
export VLLM_MLA_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1
export VLLM_TORCH_PROFILER_DIR=${PWD}/profile
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NUM_DUMMY_LAYERS=${DUMMY}
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE  # TP8+COMPOSITE | TP16+FLAT
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export XPU_CCL_BACKEND=${XPU_CCL_BACKEND}

export OPT_W8A8_BLOCK_FP8_MATMUL=1

ARGS="--model ${MODEL} \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 2048 \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP}"

[ ${EP} == 1 ] && ARGS+=" --enable-expert-parallel"


if [[ ${BENCHMARK} == "LATENCY" ]]; then
  SCRIPT="benchmarks/benchmark_latency.py"

  ARGS+=" --input-len ${IN} \
    --output-len ${OUT} \
    --batch-size ${BS} \
    --n ${BEAM} \
    --num-iters-warmup 3 \
    --num-iters 5"

  if [[ ${BEAM} != 1 ]]; then
    ARGS+=" --use-beam-search"
  fi

  if [[ ${PROFILE} == "1" ]]; then
    ARGS+=" --profile --profile-result-dir ${VLLM_TORCH_PROFILER_DIR}"
  fi
elif [[ ${BENCHMARK} == "THROUGHPUT" ]]; then
  SCRIPT="benchmarks/benchmark_throughput.py"

  ARGS+=" --input-len ${IN} \
    --output-len ${OUT} \
    --backend vllm \
    --n ${BEAM} \
    --num-prompts 50"
elif [[ ${BENCHMARK} == "SERVER" ]]; then
  SCRIPT="benchmarks/benchmark_serving.py"

  # TBD
  ARGS+=" --input-len ${IN} \
    --output-len ${OUT}"
elif [[ ${PROFILE} == "1" && ${VLLM_USE_V1} == "0" ]]; then
  SCRIPT="examples/offline_inference/profiling.py"

  ARGS+=" --save-chrome-traces-folder ./profile \
    --prompt-len ${IN} \
    run_num_steps --num-steps ${OUT}"
else
  SCRIPT="examples/offline_inference/cli.py"
fi

CMD="python ${SCRIPT} ${ARGS}"
echo CMD=${CMD}

date
eval ${CMD}
date
