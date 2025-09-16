# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from vllm.platforms import current_platform

if current_platform.is_cuda():
    from .chunk import chunk_gated_delta_rule
    from .fused_recurrent import fused_recurrent_gated_delta_rule
    from .layernorm_guard import RMSNormGated
else:
    from .torch_native_hybrid_linear_attn import (chunk_gated_delta_rule,
                                                  fused_recurrent_gated_delta_rule,
                                                  RMSNormGated)


__all__ = [
    "RMSNormGated",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
