# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.debug("Import error msg: %s", e.msg)


class ipex_ops:

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        k_scale_float: float = 1.0,
        v_scale_float: float = 1.0,
    ) -> None:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
            k_scale_float, v_scale_float)

    @staticmethod
    def flash_attn_varlen_func(
        out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,  # we don't support this in ipex kernel
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
        block_table: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        window_size: Optional[list[int]] = None,
        softcap: Optional[float] = 0.0,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        # The following parameters are not used in ipex kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        s_aux: Optional[torch.Tensor] = None,
    ):
        if cu_seqlens_k is None:
            # cu_seqlens_k is not used in ipex kernel.
            cu_seqlens_k = torch.cumsum(seqused_k, dim=0)
            cu_seqlens_k = torch.cat([
                torch.tensor([0], device=seqused_k.device, dtype=torch.int32),
                cu_seqlens_k
            ]).to(torch.int32)

        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])  # noqa: F841
        return flash_attn_varlen_func(
            out=out,
            q=q.contiguous(),
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            #alibi_slopes = alibi_slopes,
            #softcap=softcap,
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
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_k_new: Optional[torch.Tensor] = None,
            cache_leftpad: Optional[torch.Tensor] = None,
            page_size: Optional[int] = None,
            max_seqlen_k_new=0,
            causal=False,
            window_size=(-1, -1),  # -1 means infinite context window
            has_softcap=False,
            num_splits=0,  # Can be tuned for speed
            pack_gqa=None,  # Can be tuned for speed
            sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for ipex_ops, "
            "returning None.")
        return None
