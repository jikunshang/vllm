# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional

import torch
from packaging import version

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.awq import (AWQLinearMethod,
                                                         is_layer_skipped_awq)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

MIN_IPEX_VERSION = "2.6.0"


class IPEXConfig(QuantizationConfig):
    """INT8 quantization config class using IPEX for the CPU/XPU backend,
    including AWQ, GPTQ.
    """

    IPEX_QUANT_METHOD_MAP = {
        "awq": 1,
        "gptq": 0,
    }

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
        modules_to_not_convert: Optional[list[str]] = None,
        desc_act: Optional[bool] = None,
        lm_head_quantized: Optional[bool] = None,
        has_zp: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.modules_to_not_convert = modules_to_not_convert or []
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = 32 // self.weight_bits

        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits

        if self.weight_bits not in [4]:
            raise ValueError(f"IPEX quantization supports weight bits [4], "
                             f"but got {self.weight_bits}.")

        if self.method not in ["awq", "gptq", "auto-round"]:
            raise ValueError(f"IPEX quantization supports [awq, gptq], "
                             f"but got {self.method}.")

    def __repr__(self) -> str:
        return (f"IPEXConfig(method={self.method},"
                f"weight_bits={self.weight_bits}, "
                f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ipex"

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
    def from_config(cls, config: dict[str, Any]) -> "IPEXConfig":
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        if method == "awq":
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group_size = cls.get_from_keys(config,
                                           ["q_group_size", "group_size"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None)
            return cls(method, weight_bits, group_size, modules_to_not_convert,
                       False, False)
        # otherwise for gptq
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        desc_act = cls.get_from_keys_or(config, ["desc_act"], default=False)
        return cls(method, weight_bits, group_size, [], desc_act,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        if not current_platform.is_cpu() and not current_platform.is_xpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq", "gptq", "auto-round"]:
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        print(f"!!!!!prefix: {prefix} layer: {layer}")
        if isinstance(layer, LinearBase):
            if self.method == "awq":
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                    return UnquantizedLinearMethod()
                return IPEXAWQLinearMethod(self)
            if self.method == "gptq":
                return IPEXGPTQLinearMethod(self)
        if isinstance(layer, FusedMoE) and self.method == "auto-round":
            return IPEXAutoRoundFusedMoEMethod(self)
        return None


class IPEXAutoRoundFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        self.alpha = 1.702
        self.limit = 7.0

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1
        self.hidden_size = hidden_size

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or \
                hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": False
        })

        assert 'weight_loader' in extra_weight_attrs
        weight_loader = extra_weight_attrs['weight_loader']
        wrapped_weight_loader = IPEXAutoRoundFusedMoEMethod.get_weight_loader(
            layer, weight_loader)
        extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // 8,  # 8 int4 store to int32
            2 * hidden_size,
            dtype=torch.int32),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // 8, # 8 int4 store to int32
            intermediate_size_per_partition,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size // group_size + 1,  # 23
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size // group_size + 1 ,#23
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                 hidden_size,
                                                 dtype=params_dtype),
                                     requires_grad=False)
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                hidden_size // group_size + 1,
                2 * intermediate_size_per_partition // 8,
                dtype=torch.int32),
                                            requires_grad=False)
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                hidden_size // bit8_pack_factor + 1,
                intermediate_size_per_partition // group_size,
                dtype=torch.int32),
                                           requires_grad=False)
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if True:  # self.quant_config.quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys = ["w13_g_idx", "w2_g_idx"]
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(torch.empty((0, ),
                                                       dtype=torch.int32),
                                           requires_grad=False)
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)

        print(
            f"test: layer.w13_qweight shape: {layer.w13_qweight.shape}, bias shape: {layer.w13_bias.shape}, scale shape: {layer.w13_scales.shape},  zero shape: {layer.w13_qzeros.shape if self.quant_config.has_zp else None}"
        )

    def topk(self, router_logits, top_k: int):
        router_top_value, router_indices = torch.topk(
            router_logits, top_k, dim=-1)  # (num_tokens, top_k)

        router_top_value = torch.nn.functional.softmax(
            router_top_value, dim=-1, dtype=router_top_value.dtype)

        router_scores = torch.zeros_like(router_logits).scatter_(
            dim=1, index=router_indices, src=router_top_value)

        return router_scores, router_indices

    def int4_linear(self,
                    x: torch.Tensor,
                    qweight: torch.Tensor,
                    scales: torch.Tensor,
                    qzero: torch.Tensor,
                    g_idx: torch.Tensor,
                    bias: torch.Tensor,
                    block_size: int = 128) -> torch.Tensor:
        out = torch.ops.torch_ipex.mm_bias_int4(x, qweight, bias, scales, qzero,
                                           block_size, g_idx)
        return out

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        router_scores, router_indices = self.topk(router_logits, top_k)
        num_experts = router_scores.shape[1]

        batch_size = x.shape[0]
        x = x.reshape(-1, self.hidden_size)

        x = x.repeat(num_experts, 1)
        x = x.view(num_experts, -1, self.hidden_size)

        gate_ups = []
        for i in range(num_experts):
            gate_up = self.int4_linear(
                x[i], layer.w13_qweight[i], layer.w13_scales[i],
                layer.w13_qzeros[i] if self.quant_config.has_zp else None,
                layer.w13_g_idx, layer.w13_bias[i])
            gate_ups.append(gate_up)
        gate_up = torch.stack(gate_ups, dim=0)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)

        next_states = []
        for i in range(num_experts):
            next_state = self.int4_linear(
                (up[i] + 1) * glu[i], layer.w2_qweight[i], layer.w2_scales[i],
                layer.w2_qzeros[i] if self.quant_config.has_zp else None,
                layer.w2_g_idx, layer.w2_bias[i])
            next_states.append(next_state)
        next_states = torch.stack(next_states, dim=0)

        next_states = next_states.view(num_experts, batch_size, -1,
                                       self.hidden_size)
        next_states = next_states * router_logits.transpose(0, 1).view(
            num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)

        return next_states

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(param: torch.nn.Parameter,
                                    loaded_weight: torch.Tensor,
                                    weight_name: str, shard_id: str,
                                    expert_id: int):
            if "g_idx" in weight_name:
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return

            device = get_tp_group().device
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            # convert gptq and awq weight to a standard format
            if layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [4, 8]
                if "weight" in weight_name:
                    loaded_weight = loaded_weight.T.contiguous().view(
                        torch.uint8)
                elif "zeros" in weight_name:
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(
                            loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T

            # repeat the qzeros/scales to fit new group size
            if layer.group_size_div_factor > 1 and \
                    "qzeros" in weight_name or "scales" in weight_name:
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1)

            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(layer.tp_size, -1,
                                            loaded_weight.size(1))[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, :shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2:] = tensor
            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.tp_size, -1)[:, tp_rank]
            else:
                weight_loader(param, loaded_weight, weight_name, shard_id,
                              expert_id)

        return moe_wna16_weight_loader


class IPEXGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ linear method using IPEX for the CPU/XPU backend.
    """

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex
            if version.parse(
                    ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}.")
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method.") from err
        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )
        layer.ipex_output_size = layer.qweight.shape[-1]
        g_idx = layer.g_idx if self.quant_config.desc_act else None
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            g_idx=g_idx,
            bias=bias,
            group_size=self.quant_config.group_size,
            quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["gptq"]
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))


class IPEXAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method using IPEX for the CPU/XPU backend.
    """

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)

        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex
            if version.parse(
                    ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}.")
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method.") from err

        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )

        layer.ipex_output_size = layer.qweight.size(
            1) * self.quant_config.pack_factor
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            bias=bias,
            group_size=self.quant_config.group_size,
            quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["awq"]  # type: ignore
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))
