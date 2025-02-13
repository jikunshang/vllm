#!/bin/bash
set -x

# Usage: source benchmark_client_param.sh 
#        test_benchmark_client_accuracy INPUT_LEN OUTPUT_LEN MAX_CONCURRENCY NUM_PROMPTS [LEN_RATIO] [HOST] [PORT] [MODEL_PATH] [RESULTS_DIR]
# Defaults: LEN_RATIO=1.0, HOST=127.0.0.1, PORT=8688, MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}

test_benchmark_client_serving() {
  export PT_HPU_LAZY_MODE=1
  INPUT_LEN=$1
  OUTPUT_LEN=$2
  MAX_CONCURRENCY=$3
  NUM_PROMPTS=$4
  LEN_RATIO=${5:-1.0}
  HOST=${6:-127.0.0.1}
  PORT=${7:-8688}
  MODEL_PATH=${8:-${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}}
  RESULTS_DIR=${9:-logs/test-results}
  mkdir -p "$RESULTS_DIR"

  export no_proxy=localhost,${HOST},10.239.129.9

  # Run serving benchmark
  echo "Running serving benchmark: input=${INPUT_LEN}, output=${OUTPUT_LEN}, concurrency=${MAX_CONCURRENCY}, prompts=${NUM_PROMPTS}, ratio=${LEN_RATIO}"
  TIMESTAMP=$(TZ='Asia/Kolkata' date +%F-%H-%M-%S)
  LOG_BASE="benchmark_${NUM_PROMPTS}prompts_${MAX_CONCURRENCY}bs_in${INPUT_LEN}_out${OUTPUT_LEN}_ratio${LEN_RATIO}_${TIMESTAMP}"

  python3 ../benchmarks/benchmark_serving.py \
      --backend vllm \
      --model "${MODEL_PATH}" \
      --trust-remote-code \
      --host "${HOST}" \
      --port "${PORT}" \
      --dataset-name random \
      --random-input-len "${INPUT_LEN}" \
      --random-output-len "${OUTPUT_LEN}" \
      --random-range-ratio "${LEN_RATIO}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate inf \
      --seed 0 \
      --ignore-eos \
      --save-result \
      --result-filename "${RESULTS_DIR}/${LOG_BASE}.json"
}
