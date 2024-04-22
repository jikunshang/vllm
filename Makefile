.PHONY: clean build

install_deps:
	pip install wheel packaging ninja setuptools>=49.4.0 numpy
	pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

install:
	VLLM_TARGET_DEVICE=cpu pip install --no-build-isolation  -v -e .

VLLM_TP_2S_bench:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=3 ray start --head --num-cpus=32 --num-gpus=0
	cd benchmarks && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=2 python3 benchmark_throughput.py --backend=vllm --dataset=/root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/HF_models/vicuna-7b-v1.5/ --n=1 --num-prompts=1000 --dtype=bfloat16 -tp=2 --trust-remote-code --device=cpu

HF_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=hf --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/frameworks.bigdata.dev-ops/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --hf-max-batch-size=1 --trust-remote-code --device=cpu

VLLM_TP_bench:
	cd benchmarks && \
	 OMP_DISPLAY_ENV=verbose \
	 VLLM_CPU_KVCACHE_SPACE=40 \
	 python benchmark_throughput.py --backend=vllm --dataset=/root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/HF_models/vicuna-7b-v1.5/ --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code

VLLM_LT_bench:
	cd benchmarks && python benchmark_latency.py --model=/root/HF_models/vicuna-7b-v1.5/ --n=1 --batch-size=8 --input-len=128 --output-len=512 --num-iters=4 --dtype=bfloat16 --trust-remote-code --device=cpu --swap-space=40

VLLM_SERVE_bench:
	cd benchmarks && python -m vllm.entrypoints.api_server \
        --model /root/HF_models/vicuna-7b-v1.5/ --swap-space 40 \
        --disable-log-requests --dtype=bfloat16 --device cpu & \
	cd benchmarks && sleep 30 && python benchmark_serving.py \
        --backend vllm \
        --tokenizer /root/HF_models/vicuna-7b-v1.5/ --dataset /root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate 10
