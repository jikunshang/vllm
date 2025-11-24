# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch.nn.parameter import Parameter
from vllm_xpu_kernels.quantization._quantize_convert import (
    AWQUtils,
    GPTQUtils,
    transpose_onednn_woq_format,
)

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.awq import (
    AWQLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.platforms import current_platform


class XPUInt4Config(QuantizationConfig):
    """INT4 quantization config class for the XPU backend,
    including AWQ, GPTQ.
    """

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        modules_to_not_convert: list[str] | None = None,
        desc_act: bool | None = None,
        lm_head_quantized: bool | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.modules_to_not_convert = modules_to_not_convert or []
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = 32 // self.weight_bits

        if self.weight_bits not in [4]:
            raise ValueError(
                f"XPU_INT4 quantization supports weight bits [4], "
                f"but got {self.weight_bits}."
            )

        if self.method not in ["awq", "gptq"]:
            raise ValueError(
                f"XPU_INT4 quantization supports [awq, gptq], but got {self.method}."
            )

    def __repr__(self) -> str:
        return (
            f"XPUInt4Config(method={self.method},"
            f"weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "xpu_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "XPUInt4Config":
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        if method == "awq":
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
            sym = not cls.get_from_keys_or(config, ["zero_point"], default=False)
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
            return cls(
                method,
                weight_bits,
                group_size,
                sym,
                modules_to_not_convert,
                False,
                False,
            )
        elif method == "gptq":
            weight_bits = cls.get_from_keys(config, ["bits"])
            group_size = cls.get_from_keys(config, ["group_size"])
            sym = cls.get_from_keys_or(config, ["sym"], default=True)
            lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
            desc_act = cls.get_from_keys_or(config, ["desc_act"], default=False)
            return cls(
                method, weight_bits, group_size, sym, [], desc_act, lm_head_quantized
            )
        else:
            raise ValueError(
                f"XPU_INT4 quantization only supports [awq, gptq], but got {method}."
            )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        if not current_platform.is_xpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq", "gptq"]:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase):
            if self.method == "awq":
                if is_layer_skipped(prefix, self.modules_to_not_convert):
                    return UnquantizedLinearMethod()
                return XPUAWQLinearMethod(self)
            if self.method == "gptq":
                return XPUGPTQLinearMethod(self)
        return None


class XPUGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ linear method for the XPU backend."""

    def __init__(self, quant_config: XPUInt4Config):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        if self.quant_config.desc_act and layer.g_idx is not None:
            gptq_utils = GPTQUtils(bits=4, blocksize=self.quant_config.group_size)
            qweight_new, g_idx_new = gptq_utils.shuffle(layer.qweight, layer.g_idx)
            layer.qweight.data.copy_(qweight_new)
            layer.g_idx.data.copy_(g_idx_new)
            qweight_new = None
            g_idx_new = None
            del qweight_new, g_idx_new
        transpose_onednn_woq_format(layer, "gptq", self.quant_config.sym)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            layer.quant_config.group_size,
            None,
        )
        return out


class XPUAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method for the XPU backend."""

    def __init__(self, quant_config: XPUInt4Config):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)
        layer.xpu_output_size = layer.qweight.size(1) * self.quant_config.pack_factor
        qweight_new, qzeros_new = AWQUtils.repack(layer.qweight, layer.qzeros)
        if qweight_new.shape != layer.qweight.data.shape:
            layer.qweight.data = layer.qweight.data.view_as(qweight_new)
        if qzeros_new.shape != layer.qzeros.data.shape:
            layer.qzeros.data = layer.qzeros.data.view_as(qzeros_new)
        layer.qweight.data.copy_(qweight_new)
        layer.qzeros.data.copy_(qzeros_new)
        transpose_onednn_woq_format(layer, "awq", self.quant_config.sym)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            self.quant_config.group_size,
            None,
        )
        out_shape = x.shape[:-1] + (layer.xpu_output_size,)
        return out.reshape(out_shape)
