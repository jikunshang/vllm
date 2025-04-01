export MAX_MODEL_LEN=8192
# export MODEL_PATH=/software/data/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/
export MODEL_PATH=/mnt/jarvis1-disk3/HF_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8200 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 8 --max-num-seqs 32  --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
