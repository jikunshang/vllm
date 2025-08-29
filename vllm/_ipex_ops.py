# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.debug("Import error msg: %s", e.msg)


class ipex_ops:
    @staticmethod
    def rms_norm(
        input: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        out = torch.empty_like(input)
        torch.ops.torch_ipex.rms_norm_vllm(out, input.contiguous(), weight, epsilon)
        return out

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        k_scale_float: float = 1.0,
        v_scale_float: float = 1.0,
    ) -> None:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale_float,
            v_scale_float,
        )

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
        seqused_k: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        # passed in qwen vl
        dropout_p: float = 0.0,
        # The following parameters are not used in ipex kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        s_aux: torch.Tensor | None = None,
    ):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])

        if block_table is None:
            assert cu_seqlens_k is not None, (
                "cu_seqlens_k can't be None when calling varlen_attention."
            )
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            ipex_ops.varlen_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale,
                False,
                causal,
                False,
                None,
                real_window_size[0],
                real_window_size[1],
                -1,
            )
            return out
        else:
            return ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
                out,
                q.contiguous(),
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
                block_table,
                alibi_slopes,
                sink=s_aux,
                softcap=softcap,
                window_size_left=real_window_size[0],
                window_size_right=real_window_size[1],
                k_scale=1.0,
                v_scale=1.0,
            )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        page_size: int | None = None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        has_softcap=False,
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for ipex_ops, returning None."
        )
        return None
