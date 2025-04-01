#export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/
export MODEL_PATH=/mnt/jarvis1-disk3/HF_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill 10.239.15.51:8100 \
    --decode 10.239.15.50:8200 \
    --port 8868
