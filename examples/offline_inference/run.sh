#!/bin/bash

: ${MODEL=${1:-"deepseek-ai/DeepSeek-R1"}} # HF model id
: ${TP=${2:-8}}                            # tensor-parallel
: ${EP=${4:-1}}                            # expert-parallel - (MOE)
: ${PP=${3:-1}}                            # pipeline-parallel - (AsyncLLMEngine)-WIP
: ${DP=${5:-1}}                            # data-parallel - (MLA)-WIP
: ${DUMMY=${6:--1}}                        # number dummy layers - default -1 to disable dummy weight
: ${IN=${7:-1024}}                         # input prompt length
: ${OUT=${8:-4}}                           # output generate length

export VLLM_USE_V1=1
export VLLM_MLA_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NUM_DUMMY_LAYERS=${DUMMY}
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE  # TP8+COMPOSITE | TP16+FLAT
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

export OPT_GROUPED_TOPK_SIGMOID=1
export OPT_W8A8_BLOCK_FP8_MATMUL=1

CMD="python cli.py"

ARGS="--model ${MODEL} \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 2048 \
  --input-len ${IN} \
  --output-len ${OUT} \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP}"
[ ${EP} == 1 ] && CMD+=" --enable-expert-parallel"

echo CMD=${CMD}
eval ${CMD}
