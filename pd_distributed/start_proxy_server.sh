export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py --model $MODEL_PATH --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200  --port 8123