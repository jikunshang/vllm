#!/bin/bash
VLLM_USE_V1=0 VLLM_DP_RANK=0 VLLM_DP_SIZE=2 CUDA_VISIBLE_DEVICES="0,1" VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 python -m vllm.entrypoints.openai.api_server --model 'ibm-research/PowerMoE-3b' --port 8801 --tensor-parallel-size 2 &

VLLM_USE_V1=0 VLLM_DP_RANK=1 VLLM_DP_SIZE=2 CUDA_VISIBLE_DEVICES="2,3" VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 python -m vllm.entrypoints.openai.api_server --model 'ibm-research/PowerMoE-3b' --port 8802 --tensor-parallel-size 2

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
