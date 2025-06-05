# SPDX-License-Identifier: Apache-2.0
import torch

from vllm import _custom_ops as ops

a = torch.randn(2, 3, 4, 5, device="xpu")

ops.hello_vllm(a)
