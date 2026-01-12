# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4MxFp4"]


class CompressedTensorsW4A4MxFp4(CompressedTensorsScheme):
    def __init__(self):
        self.backend = "none"

        logger.info_once(f"Using {self.backend} for MXFP4 GEMM")
        self.group_size = 32  # MXFP4 uses block size of 32

    @classmethod
    def get_min_capability(cls) -> int:
        # MXFP4 support typically requires SM_80+
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight (packed FP4)
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale (MXFP4 uses E8M0 format for scales)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,  # Using float8_e4m3fn to represent E8M0
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        # Input Global Scale
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Quantize input to MXFP4
        # For MXFP4, we need to quantize with MX (microscaling) format
        # This involves per-block scaling with E8M0 shared exponents
        try:
            from vllm._custom_ops import mxfp4_gemm_ref, mxfp4_quant_ref
        except ImportError as exc:
            raise ImportError(
                "mxfp4_quant_ref is required for MXFP4 quantization"
            ) from exc
        orig_dtype = x.dtype
        x_fp4, x_blockscale = mxfp4_quant_ref(x)

        out = mxfp4_gemm_ref(
            x_fp4,
            layer.weight_packed,
            x_blockscale,
            layer.weight_scale,
            orig_dtype,
        )

        # if bias is not None:
        #     out = out + bias
        return out
