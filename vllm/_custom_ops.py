# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType

logger = init_logger(__name__)

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
    )


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v2(
        out,
        exp_sum,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
    )


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    mfma_type: str = "fp8" if envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN else "f16",
) -> None:
    torch.ops._rocm_C.paged_attention(
        out,
        exp_sum,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        query_start_loc,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        fp8_out_scale,
        mfma_type,
    )


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cpu.mla_decode_kvcache(
        out, query, kv_cache, scale, block_tables, seq_lens
    )


# merge attn states ops
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    torch.ops._C.merge_attn_states(
        output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse
    )


def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops._C.convert_vertical_slash_indexes(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops._C.convert_vertical_slash_indexes_mergehead(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        vertical_indices_count,
        slash_indices_count,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


# layer norm ops
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    # If removed, also need to remove contiguous in MatcherRMSNorm
    # Kunshang: revert back #28103.
    input_contiguous = input.contiguous()
    torch.ops._C.rms_norm(out, input_contiguous, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
) -> None:
    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
    )


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    torch.ops._C.apply_repetition_penalties_(
        logits, prompt_mask, output_mask, repetition_penalties
    )


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )

    torch.ops._C.rms_norm_dynamic_per_token_quant(
        output, input, weight, scales, epsilon, scale_ub, residual
    )
    return output, scales


# fused quant layer norm ops blocked
def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list[int],
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(group_size) == 2
    output = torch.empty_like(input, dtype=quant_dtype)
    if is_scale_transposed:
        scales = torch.empty(
            (input.shape[-1] // group_size[1], input.numel() // input.shape[-1]),
            device=input.device,
            dtype=torch.float32,
        ).transpose(0, 1)
    else:
        scales = torch.empty(
            (input.numel() // input.shape[-1], input.shape[-1] // group_size[1]),
            device=input.device,
            dtype=torch.float32,
        )

    torch.ops._C.rms_norm_per_block_quant(
        output,
        input,
        weight,
        scales,
        epsilon,
        scale_ub,
        residual,
        group_size[1],
        is_scale_transposed,
    )
    return output, scales


# quantization ops
# awq
def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton,
        )

        return awq_dequantize_triton(qweight, scales, zeros)
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy)


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton

        return awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, scales, qzeros, split_k_iters)


# gptq
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    use_v2_format: bool,
    bit: int,
) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        use_exllama,
        use_v2_format,
        bit,
    )


if hasattr(torch.ops._C, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_gptq_qzeros: torch.Tensor,
        b_gptq_scales: torch.Tensor,
        b_g_idx: torch.Tensor,
        use_exllama: bool,
        use_v2_format: bool,
        bit: int,
    ) -> torch.Tensor:
        return torch.empty(
            (a.size(0), b_q_weight.size(1)), dtype=a.dtype, device=a.device
        )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


# marlin_24
def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_24_gemm(
        a, b_q_weight, b_meta, b_scales, workspace, b_q_type.id, size_m, size_n, size_k
    )


if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):

    @register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_meta: torch.Tensor,
        b_scales: torch.Tensor,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type_id: int,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor:
        dtype = a.dtype
        if dtype not in [torch.half, torch.bfloat16]:
            dtype = b_scales.dtype
        return torch.empty((size_m, size_n), device=a.device, dtype=dtype)

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        split_k_iters: torch.SymInt,
        thx: int,
        thy: int,
    ) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c), dtype=scales.dtype, device=scales.device)

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        split_k_iters: torch.SymInt,
    ) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty(
            (split_k_iters, num_in_feats, qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        ).sum(0)

    @register_fake("_C::machete_mm")
    def machete_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: torch.dtype | None = None,
        b_group_scales: torch.Tensor | None = None,
        b_group_zeros: torch.Tensor | None = None,
        b_group_size: int | None = None,
        b_channel_scales: torch.Tensor | None = None,
        a_token_scales: torch.Tensor | None = None,
        schedule: str | None = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
        b_q_weight: torch.Tensor,
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: torch.dtype | None,
    ) -> torch.Tensor:
        return torch.empty_like(b_q_weight, memory_format=torch.contiguous_format)

    @register_fake("_C::cutlass_w4a8_mm")
    def cutlass_w4a8_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
        b_q: torch.Tensor,
        b_group_scales: torch.Tensor,
        b_group_size: int,
        b_channel_scales: torch.Tensor,
        a_token_scales: torch.Tensor,
        out_type: torch.dtype | None = None,
        maybe_schedule: str | None = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        out_dtype = out_type if out_type is not None else torch.bfloat16
        return torch.empty((m, n), device=a.device, dtype=out_dtype)

    @register_fake("_C::cutlass_pack_scale_fp8")
    def cutlass_pack_scale_fp8_fake(scales: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(scales, memory_format=torch.contiguous_format)

    @register_fake("_C::cutlass_encode_and_reorder_int4b")
    def cutlass_encode_and_reorder_int4b_fake(b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(b, memory_format=torch.contiguous_format)

    @register_fake("_C::cutlass_encode_and_reorder_int4b_grouped")
    def cutlass_encode_and_reorder_int4b_grouped_fake(b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(b, memory_format=torch.contiguous_format)


if hasattr(torch.ops._C, "allspark_w8a16_gemm"):

    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(
        a: torch.Tensor,
        b_qweight: torch.Tensor,
        b_scales: torch.Tensor,
        b_qzeros: torch.Tensor | None,
        n: torch.SymInt,
        group_size: torch.SymInt,
        sm_count: torch.SymInt,
        sm_version: torch.SymInt,
        CUBLAS_M_THRESHOLD: torch.SymInt,
        has_zp: bool,
        n32k16_reorder: bool,
    ) -> torch.Tensor:
        m = a.size(0)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row), dtype=torch.float16, device=W.device)


if hasattr(torch.ops._C, "ggml_moe_a8_vec"):

    @register_fake("_C::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row), dtype=X.dtype, device=W.device)


# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if current_platform.is_rocm() or not cutlass_compatible_b:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj, azp, bias)
    return out.view(*target_shape)


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(cuda_device_capability)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    try:
        return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)
    except AttributeError:
        # Return False on non-CUDA platforms where it is not available
        return False


def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    assert a.dtype in [torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16]
    assert a.is_contiguous()

    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    assert a.shape[1] % (2 * elemsPerMetaElem) == 0

    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
    a: torch.Tensor,
    bt_nzs: torch.Tensor,
    bt_meta: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    assert bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] and bias.dtype == out_dtype

    m = a.shape[0]
    n = bt_nzs.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_sparse_mm(
        out, a, bt_nzs, bt_meta, scale_a, scale_b, bias
    )

    return out


def get_cutlass_moe_mm_data(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: torch.Tensor | None = None,
):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    - blockscale_offsets: Optional argument passed for fp4 moe. Indices that
                          mark at which block scale index each expert begins
                          its computation. The number of block scale rows
                          computed with expert E is blockscale_offsets[E + 1] -
                          blockscale_offsets[E]
    """
    return torch.ops._C.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )


def get_cutlass_moe_mm_problem_sizes(
    topk_ids: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: torch.Tensor | None = None,
    force_swap_ab: bool | None = None,
):
    """
    Compute only the per-expert problem sizes needed by the two grouped matrix
    multiplications used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token→expert mapping) and computes:
    - problem_sizes1, problem_sizes2: M×N×K sizes of each expert's
                                    multiplication for the two grouped MMs
                                    used in the fused MoE operation.
    Optional:
    - force_swap_ab: If set to True or False, explicitly enable or disable the
                     A/B input swap optimization. If None (default), the swap
                     is selected automatically based on tensor sizes.
    """
    return torch.ops._C.get_cutlass_moe_mm_problem_sizes(
        topk_ids,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n,
        k,
        blockscale_offsets,
        force_swap_ab,
    )


def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor):
    """
    Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor.
    This is used in MoE to permute the input tensor before performing grouped matrix multiplications.
    """
    num_tokens_permuted = dst2src_map.shape[0]
    output_tensor = torch.empty(
        (num_tokens_permuted, input_tensor.shape[1]),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops._moe_C.shuffle_rows(input_tensor, dst2src_map, output_tensor)
    return output_tensor


def get_cutlass_pplx_moe_mm_data(
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    num_local_experts: int,
    padded_m: int,
    n: int,
    k: int,
):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in expert_num_tokens (token count per expert) and
    non_zero_expert_idxs (consecutive indices of experts with non-zero token
    counts) and uses them to compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation.
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    """
    return torch.ops._C.get_cutlass_pplx_moe_mm_data(
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        expert_num_tokens,
        num_local_experts,
        padded_m,
        n,
        k,
    )


def cutlass_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    return torch.ops._C.cutlass_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        per_act_token,
        per_out_ch,
    )


def cutlass_fp4_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alphas: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    sf_offsets: torch.Tensor,
):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    return torch.ops._C.cutlass_fp4_group_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        alphas,
        problem_sizes,
        expert_offsets,
        sf_offsets,
    )


# gptq_marlin
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_repack(
        b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit
    )


if hasattr(torch.ops._C, "gptq_marlin_repack"):

    @register_fake("_C::gptq_marlin_repack")
    def _gptq_marlin_repack_fake(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: torch.SymInt,
        size_n: torch.SymInt,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        marlin_tile_size = 16
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


# awq_marlin
def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    return torch.ops._C.awq_marlin_repack(
        b_q_weight, size_k, size_n, num_bits, is_a_8bit
    )


if hasattr(torch.ops._C, "awq_marlin_repack"):

    @register_fake("_C::awq_marlin_repack")
    def _awq_marlin_repack_fake(
        b_q_weight: torch.Tensor,
        size_k: torch.SymInt,
        size_n: torch.SymInt,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        marlin_tile_size = 16
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = torch.ops._C.gptq_marlin_repack(
            b_q_weight[e], perm[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = torch.ops._C.awq_marlin_repack(
            b_q_weight[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False,
):
    return torch.ops._C.marlin_int4_fp8_preprocess(qweight, qzeros_or_none, inplace)


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


# machete
def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
    group_zeros_type: torch.dtype | None = None,
    channel_scales_type: torch.dtype | None = None,
    token_scales_type: torch.dtype | None = None,
    out_type: torch.dtype | None = None,
) -> list[str]:
    return torch.ops._C.machete_supported_schedules(
        a_type,
        b_type.id,
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by machete_prepack_B
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor:
    return torch.ops._C.machete_mm(
        a,
        b_q,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


def machete_prepack_B(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(
        b_q_weight, a_type, b_type.id, group_scales_type
    )


# CUTLASS W4A8
def cutlass_w4a8_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
    b_q: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    b_channel_scales: torch.Tensor,
    a_token_scales: torch.Tensor,
    out_type: torch.dtype | None = None,
    maybe_schedule: str | None = None,
) -> torch.Tensor:
    return torch.ops._C.cutlass_w4a8_mm(
        a,
        b_q,
        b_group_scales,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        out_type,
        maybe_schedule,
    )


def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_pack_scale_fp8(scales)


def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_encode_and_reorder_int4b(b)


def cutlass_w4a8_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    group_scale_strides: torch.Tensor,
    maybe_schedule: str | None = None,
):
    """
    Executes the CUTLASS-based fused-MoE grouped matrix multiplication for the
    W4A8 quantization scheme. Uses group-wise quantization (INT4 -> FP8)
    and both per-channel + per-token scaling in the epilogue.

    Args:
        out_tensors:
            Output buffer for all experts (updated in-place).
        a_tensors:
            FP8 (E4M3FN) activations for all experts.
        b_tensors:
            INT4-packed weight matrix for all experts, packed to INT32
        a_scales:
            Per-token FP8 activation scales, applied in the epilogue.
        b_scales:
            Per-channel FP8 weight scales for each expert, applied in the epilogue.
        b_group_scales:
            FP8 scale values for group-wise INT4 weight blocks.
        b_group_size:
            Number of elements grouped under each entry of b_group_scales.
        expert_offsets:
            Cumulative token offsets
        problem_sizes:
            Per-expert (M, N, K) GEMM sizes used by the grouped GEMM launcher.
        a/b/c/group_scale_strides:
            Strides describing the memory layout of the input tensors.
        maybe_schedule:
            Optional override to choose a specific kernel or epilogue schedule.

    Returns:
        out_tensors updated in-place with the dequantized INT4xFP8 grouped GEMM result.
    """
    return torch.ops._C.cutlass_w4a8_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        b_group_scales,
        b_group_size,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule,
    )


def cutlass_encode_and_reorder_int4b_grouped(
    b_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.cutlass_encode_and_reorder_int4b_grouped(b_tensors)


if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.permute_cols(a, perm)


# mxfp4
def mxfp4_quant_ref_AAA(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP4 and return quantized tensor and scale.
    This function quantizes the last dimension of the given tensor `input`. For
    every 32 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is stored in float8_e4m3fn format.

    Args:
        input: The input tensor to be quantized to MXFP4
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in MXFP4 but every
            two values are packed into a uint8 and float8_e4m3fn scaling factors.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 32
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 32, but got {n}."
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )

    # Two mxfp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # The scales are stored in float8_e4m3fn format.
    output_scale = torch.empty((m, n // block_size), device=device, dtype=torch.uint8)

    return output, output_scale


def mxfp4_quant_ref_bbb(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise a FP16/BF16 matrix to MXFP4 (float4_e2m1fn_x2) + E8M0 scales.

    Args
    ----
    x : torch.Tensor
        shape (M, N), dtype torch.float16 or torch.bfloat16

    Returns
    -------
    q : torch.Tensor
        shape (M, N//2), dtype torch.uint8
        Each byte packs two 4-bit values (lower nibble = left element,
        upper nibble = right element) in a 2×1 micro-tile.
    s : torch.Tensor
        shape (M, N//32), dtype torch.uint8 (E8M0 exponent)
    """
    assert x.ndim == 2
    M, N = x.shape
    assert N % 32 == 0, "N must be a multiple of 32"
    assert x.dtype in (torch.float16, torch.bfloat16)

    # ------------------------------------------------------------------
    # 1.  compute abs-max per 32-column tile  ->  scale candidate
    # ------------------------------------------------------------------
    x_view = x.view(M, N // 32, 32)  # (M, N//32, 32)
    amax = x_view.abs().amax(dim=-1, keepdim=True)  # (M, N//32, 1)
    if x.size(0) != 256:
        print(f"[QUANT] amax: {amax}")

    # ------------------------------------------------------------------
    # 2.  convert abs-max to E8M0 exponent  (clamp + round)
    # ------------------------------------------------------------------
    #   scale = 2^(e-127)  =>  e = 127 + log2(scale)
    #   clamp to valid uint8 range
    # Handle zero scales
    amax_safe = torch.where(
        amax > 0, amax, torch.tensor(1.0, device=amax.device, dtype=amax.dtype)
    )
    log2 = torch.log2(amax_safe.to(torch.float32))
    e = (log2 + 127.0).round().clamp(0, 255).to(torch.uint8)
    # Set exponent to 0 for zero blocks
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    s = e.squeeze(-1)  # (M, N//32)

    # ------------------------------------------------------------------
    # 3.  de-quant scale -> FP32 for division
    # ------------------------------------------------------------------
    scale_fp32 = torch.ldexp(
        torch.ones_like(e, dtype=torch.float32), e.to(torch.int32) - 127
    )  # (M, N//32, 1)
    scale_fp32 = torch.where(amax > 0, scale_fp32, torch.ones_like(scale_fp32))
    if x.size(0) != 256:
        torch.set_printoptions(threshold=1000)
        print(f"[QUANT] scale_fp32: {scale_fp32}")

    # ------------------------------------------------------------------
    # 4.  normalise & quant to 4-bit E2M1FN
    # ------------------------------------------------------------------
    x_norm = x_view.to(torch.float32) / scale_fp32  # (M, N//32, 32)
    if x.size(0) != 256:
        print(f"[QUANT] x_norm: {x_norm}")
    x_norm = x_norm.clamp(-6.0, 6.0)  # keep in range
    if x.size(0) != 256:
        print(f"[QUANT] x_clamp: {x_norm}")
    # E2M1FN encoding: [sign][2-bit exp][1-bit mantissa]
    # Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    # Bit layout: SEEM where S=sign, EE=exp, M=mantissa

    sign = (x_norm < 0).to(torch.int32)  # 0 or 1
    aval = x_norm.abs()

    # Handle zero explicitly
    is_zero = aval < 1e-10

    # Compute exponent (log2 of absolute value)
    # For E2M1FN: exp_bias = 1, so exp ranges from -1 to 2
    exp_unbias = (
        torch.floor(torch.log2(aval.clamp(min=1e-10))).clamp(-1, 2).to(torch.int32)
    )

    # Compute mantissa: check if normalized value >= 1.5
    # normalized = aval / 2^exp should be in [1.0, 2.0)
    # mantissa bit is 1 if normalized >= 1.5
    normalized = aval / torch.pow(2.0, exp_unbias.to(torch.float32))
    mantissa = (normalized >= 1.5).to(torch.int32)  # 0 or 1

    # Encode: [sign(1bit)][exp+1(2bits)][mantissa(1bit)]
    exp_biased = (exp_unbias + 1).clamp(0, 3)  # Map -1,0,1,2 -> 0,1,2,3
    idx = (sign << 3) | (exp_biased << 1) | mantissa

    # Set zero values to 0
    idx = torch.where(is_zero, torch.zeros_like(idx), idx)
    idx = idx.clamp(0, 15).to(torch.uint8)  # 4-bit index

    # ------------------------------------------------------------------
    # 5.  pack two 4-bit indices into one uint8
    # ------------------------------------------------------------------
    idx = idx.view(M, N)  # back to (M, N)
    even = idx[:, 0::2]
    odd = idx[:, 1::2]
    q = even | (odd << 4)  # (M, N//2)
    if x.size(0) != 256:
        print(f"[QUANT] q shape: {q.shape}, dtype: {q.dtype}")
    return q, s


FP4_EBITS, FP4_MBITS = 2, 1


# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


# copy-pasted from
# https://github.com/pytorch/ao/blob/29488018d99af7f7339f06353c6b5bbeae8a1493/torchao/prototype/custom_fp_utils.py#L147
def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.
    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp - (1 << i)) << (
                    left_shift + MBITS_F32 - mbits
                )
                exp_biased_f32 = (denormal_exp_biased - left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s) for |: 'SymInt' and 'int'
                # thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32
                )

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def from_blocked_format(x_mxfp8, scales_unswizzled, blocksize=32):
    # expand scales
    scales = torch.repeat_interleave(scales_unswizzled, blocksize, dim=1)

    # de-scale and convert
    x_f32 = x_mxfp8.to(torch.float) * scales.to(torch.float)
    return x_f32.to(torch.bfloat16)


def unpack_uint4(uint8_data) -> torch.Tensor:
    # Take a packed uint8 tensor (i.e. nvfp4) and unpack into
    # a tensor twice as wide. Useful for dequant operations.
    shape = list(uint8_data.shape)
    # 2x packed elements -> single non-packed => adjust shape
    shape[-1] *= 2
    out = torch.empty(*shape, device=uint8_data.device, dtype=torch.uint8).view(-1)

    uint8_data_as_uint8 = uint8_data.view(torch.uint8).view(-1)

    out[1::2] = uint8_data_as_uint8[:] >> 4
    out[::2] = uint8_data_as_uint8 & 15

    return out.view(shape)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).
    Code below is an adaptation of https://fburl.com/code/ciwofcg4
    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


def mxfp4_quant_ref(
    data_hp: torch.Tensor,
    block_size: int = 32,
    format: str = "mxfp4",
):
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert data_hp.shape[-1] % block_size == 0, (
        f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
    )  # noqa: E501
    assert data_hp.is_contiguous(), "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if format == "mxfp8":
        F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        max_pos = F8E4M3_MAX
    elif format == "mxfp4":
        F4E2M1_MAX = 6.0
        max_pos = F4E2M1_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping # noqa: E501
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)),
                    min=-E8M0_EXPONENT_BIAS,
                    max=E8M0_EXPONENT_BIAS,
                )
                + E8M0_EXPONENT_BIAS
            ).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    if format == "mxfp8":
        data_lp = data_lp.to(torch.float8_e4m3fn)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif format == "mxfp4":
        data_lp = _bfloat16_to_float4_e2m1fn_x2(data_lp.to(torch.bfloat16))
        final_shape = list(orig_shape)
        final_shape[-1] //= 2
        data_lp = data_lp.reshape(final_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return data_lp, scale_e8m0_biased


def e8m0_to_fp32(e8m0: torch.Tensor) -> torch.Tensor:
    """
    Convert E8M0 (8-bit exponent-only) tensor to FP32.

    Args:
        e8m0: Tensor of dtype torch.uint8, shape arbitrary

    Returns:
        fp32_tensor: Tensor of dtype torch.float32
    """
    assert e8m0.dtype == torch.uint8, "Input must be uint8 (E8M0)"

    # Convert exponent to int32 to avoid overflow
    exponent = e8m0.to(torch.int32)

    # Compute 2^(exponent - 127) using ldexp
    fp32 = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)

    return fp32


def convert_f4_to_fp16(packed_tensor, target_dtype=torch.float16):
    """
    Converts a packed uint8 tensor (containing float4_e2m1fn_x2) to FP16/BF16.
    Assumes packed_tensor is uint8.
    """
    # 1. Create the Dequantization Table
    # There are only 16 possible values for an e2m1fn float.
    # These values are defined by the OCP (Open Compute Project) Microscaling spec.
    # Mapping for e2m1fn (approximate values):
    pack_int8 = packed_tensor.view(torch.uint8)
    packed_tensor_fp4 = packed_tensor.view(torch.float4_e2m1fn_x2)
    f4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,  # Positive values
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,  # Negative values
        ],
        dtype=target_dtype,
        device=packed_tensor.device,
    )

    # 2. Unpack the bits
    # Extract lower 4 bits and upper 4 bits
    val_lo = pack_int8 & 0x0F
    val_hi = (pack_int8 >> 4) & 0x0F

    # 3. Reshape and Map
    # Flatten and use as indices into our LUT
    unpacked = torch.stack([val_lo, val_hi], dim=-1).flatten().to(packed_tensor.device)
    return f4_values[unpacked.long()].view(packed_tensor_fp4.size(0), -1)


def mxfp4_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    `mxfp4_gemm_ref` implements a reference version of
        `output = torch.mm((a_scales * a), (b_scales * b)).to(out_dtype)`
    where a_scales * a and b_scales * b are implemented using numpy-style
    broadcasting.

    Args:
        a: The input tensor a in MXFP4 format.
        b: The input tensor b in MXFP4 format.
        a_scales: A tensor containing the scaling factors for tensor a.
        b_scales: A tensor containing the scaling factors for tensor b.
        out_dtype: The desired output data type (torch.bfloat16 or torch.float16).

    Returns:
        torch.Tensor: The result of the matrix multiplication in the specified output data type.
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    output_shape = (*a.shape[:-1], b.shape[0])
    out = torch.empty(output_shape, dtype=out_dtype, device=a.device)

    a_final = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(a), FP4_EBITS, FP4_MBITS),
        scales_unswizzled=a_scales,
        blocksize=32,
    )

    b_final = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(b), FP4_EBITS, FP4_MBITS),
        scales_unswizzled=b_scales,
        blocksize=32,
    )

    # if a.size(0) != 256:
    #     print(f"a_final shape: {a_final.shape} {a_final}")
    #     print(f"a_scales shape: {a_scales.shape}, {a_scales}")
    #     print(f"a shape:{a.shape}, {a}")

    # if a.size(0) != 256:
    #     print(f"b shape:{b.shape}, {b}")
    #     print(f"b_scales shape: {b_scales.shape},  {b_scales}")
    #     print(f"b_final shape: {b_final.shape} {b_final}")

    out = torch.matmul(a_final, b_final.transpose(-2, -1)).to(out_dtype)

    return out


# fp4
def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert not current_platform.is_rocm()
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    torch.ops._C.scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input_tensor: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in FP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert not current_platform.is_rocm()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    m_numtopk, k = input_tensor.shape

    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    torch.ops._C.scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.zeros((1, 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        assert scale.numel() == 1, f"{scale.shape}"
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale


# gptq allspark
def allspark_repack_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    has_zp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] :
            rearranged weight, scale, and optionally zero_point.
    """
    K = qweight.shape[0]
    N = qweight.shape[1]
    N_32align = (N + 32 - 1) // 32 * 32

    qweight_reorder = torch.empty(
        (N_32align, K), device=qweight.device, dtype=qweight.dtype
    )
    scale_reorder = torch.empty((1, N_32align), device=scale.device, dtype=scale.dtype)
    zero_point_reorder = None
    if has_zp:
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization."
        )
        zero_point_reorder = torch.empty(
            (1, N_32align), device=zero_point.device, dtype=zero_point.dtype
        )

    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight,
        scale,
        zero_point,
        has_zp,
        qweight_reorder,
        scale_reorder,
        zero_point_reorder,
        K,
        N,
        N_32align,
    )

    return qweight_reorder, scale_reorder, zero_point_reorder


def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    CUBLAS_M_THRESHOLD: int,
    has_zp: bool,
    n32k16_reorder: bool,
) -> torch.Tensor:
    return torch.ops._C.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        group_size,
        sm_count,
        sm_version,
        CUBLAS_M_THRESHOLD,
        has_zp,
        n32k16_reorder,
    )


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(
        output, input.contiguous(), input_scales, input_azp
    )
    return output, input_scales, input_azp


# gguf
def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: torch.SymInt,
    tokens: torch.SymInt,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    pad_slot_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
):
    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D_,
        z_,
        delta_bias_,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )


# ROCm skinny gemms
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor:
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


def wvSplitK(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)


def wvSplitKQ(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    cu_count: int,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    out = torch.empty((b.shape[0], a.shape[0]), dtype=out_dtype, device=b.device)
    torch.ops._rocm_C.wvSplitKQ(a, b, bias, out, scale_a, scale_b, cu_count)
    return out


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    torch.ops._moe_C.moe_sum(input, output)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        # FIXME: add this back when we support this in the sycl kernel code
        # expert_map,
    )


def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    torch.ops._moe_C.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        expert_map,
    )


def moe_wna16_gemm(
    input: torch.Tensor,
    output: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    bit: int,
) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only available on CUDA platforms"
        )
    torch.ops._moe_C.moe_wna16_gemm(
        input,
        output,
        b_qweight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        top_k,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        bit,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    torch.ops._moe_C.topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
    )


def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """
    Perform grouped top-k routing for mixture of experts.

    Args:
        scores: Raw inputs (logits if scoring_func=1, scores if scoring_func=0)
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the output weights
        routed_scaling_factor: Scaling factor for routing weights
        bias: Bias tensor (e_score_correction_bias). Always fused in kernel.
        scoring_func: 0=none (no activation), 1=sigmoid
    """
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The fused grouped_topk kernel is only available on CUDA platforms"
        )
    return torch.ops._moe_C.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
) -> torch.Tensor:
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_qzeros,
        g_idx,
        perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        thread_k,
        thread_n,
        blocks_per_sm,
    )


if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(
        a: torch.Tensor,
        b_q_weights: torch.Tensor,
        sorted_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        b_scales: torch.Tensor,
        b_zero_points: torch.Tensor,
        g_idx: torch.Tensor,
        perm: torch.Tensor,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool,
        num_experts: int,
        topk: int,
        moe_block_size: int,
        replicate_input: bool,
        apply_weights: bool,
    ) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n), dtype=a.dtype, device=a.device)

    @register_fake("_moe_C::moe_wna16_marlin_gemm")
    def moe_wna16_marlin_gemm_fake(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        is_ep: bool,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
    ):
        return torch.empty(
            (size_m * top_k, size_n), dtype=input.dtype, device=input.device
        )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


def swap_blocks(
    src: torch.Tensor, dst: torch.Tensor, block_mapping: torch.Tensor
) -> None:
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(
    output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )


def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
) -> None:
    """Gather and upconvert FP8 KV cache to BF16 workspace.

    Args:
        src_cache: FP8 KV cache [num_blocks, block_size, 656]
        dst: BF16 output workspace [total_tokens, 576]
        block_table: Block indices [num_reqs, max_blocks]
        seq_lens: Sequence lengths [num_reqs]
        workspace_starts: Workspace start offsets [num_reqs]
        batch_size: Number of requests
    """
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache, dst, block_table, seq_lens, workspace_starts, batch_size
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
    )


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device
    )


# custom ar
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(
        ipc_tensors, rank_data, rank, fully_connected
    )


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(
    fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce
def init_custom_qr(rank: int, world_size: int, qr_max_size: int | None = None) -> int:
    return torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)


def qr_destroy(fa: int) -> None:
    torch.ops._C_custom_ar.qr_destroy(fa)


def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None:
    torch.ops._C_custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)


def qr_get_handle(fa: int) -> torch.Tensor:
    return torch.ops._C_custom_ar.qr_get_handle(fa)


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    return torch.ops._C_custom_ar.qr_open_handles(fa, handles)


def qr_max_size() -> int:
    return torch.ops._C_custom_ar.qr_max_size()


def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._C.get_flash_mla_metadata(
        cache_seqlens, num_heads_per_head_k, num_heads_k
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def sm100_cutlass_mla_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    num_kv_splits: int,
) -> torch.Tensor:
    torch.ops._C.sm100_cutlass_mla_decode(
        out,
        lse,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        scale,
        num_kv_splits,
    )
    return out


def sm100_cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int, num_kv_splits: int
) -> int:
    return torch.ops._C.sm100_cutlass_mla_get_workspace_size(
        max_seq_len, num_batches, sm_count, num_kv_splits
    )


if hasattr(torch.ops._C, "weight_packed_linear"):

    @register_fake("_C::weight_packed_linear")
    def weight_packed_linear_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        bias: torch.Tensor | None,
        is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty(
            (mat1.size(0), mat2.size(0)), dtype=mat1.dtype, device=mat2.device
        )


if hasattr(torch.ops._C, "fused_experts_cpu"):

    @register_fake("_C::fused_experts_cpu")
    def fused_experts_cpu_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool,
        use_int8_w8a8: bool,
        use_fp8_w8a16: bool,
        w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        block_size: list[int] | None,
        a1_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


if hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):

    @register_fake("_C::int8_scaled_mm_with_quant")
    def int8_scaled_mm_with_quant_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        scales2: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        M = mat1.size(0)
        N = mat2.size(0)
        return torch.empty((M, N), dtype=out_dtype)


class CPUDNNLGEMMHandler:
    def __init__(self) -> None:
        self.handler: int | None = None
        self.n = -1
        self.k = -1

    def __del__(self):
        if self.handler is not None:
            torch.ops._C.release_dnnl_matmul_handler(self.handler)


_supports_onednn = bool(hasattr(torch.ops._C, "create_onednn_mm_handler"))


def is_onednn_acl_supported():
    return torch.ops._C.is_onednn_acl_supported()


def create_onednn_mm(
    weight: torch.Tensor,  # [K, N]
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    handler.handler = torch.ops._C.create_onednn_mm_handler(
        weight, primitive_cache_size
    )
    return handler


def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    torch.ops._C.onednn_mm(
        output, x.reshape(-1, dnnl_handler.k), bias, dnnl_handler.handler
    )

    return output


def create_onednn_scaled_mm(
    weight: torch.Tensor,  # [K, N]
    weight_scales: torch.Tensor,
    output_type: torch.dtype,
    dynamic_quant: bool,
    use_azp: bool,
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    handler.handler = torch.ops._C.create_onednn_scaled_mm_handler(
        weight, weight_scales, output_type, dynamic_quant, use_azp, primitive_cache_size
    )
    return handler


def onednn_scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
):
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    token_num = input.numel() // input.shape[-1]
    input = input.view((token_num, input.shape[-1]))
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((token_num, 1), device=input.device, dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales, input_azp)
    return output, input_scales, input_azp


def onednn_scaled_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor | None,
    input_zp: torch.Tensor | None,
    input_zp_adj: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    torch.ops._C.onednn_scaled_mm(
        output, x, input_scale, input_zp, input_zp_adj, bias, dnnl_handler.handler
    )

    return output


def cpu_attn_get_scheduler_metadata(
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool,
    sliding_window_size: int,
    isa: str,
    enable_kv_split: bool,
) -> torch.Tensor:
    sheduler_metadata = torch.ops._C.get_scheduler_metadata(
        num_reqs,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        dtype,
        query_start_loc,
        causal,
        sliding_window_size,
        isa,
        enable_kv_split,
    )
    return sheduler_metadata


def cpu_attn_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str,
) -> None:
    torch.ops._C.cpu_attn_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        isa,
    )


def cpu_attention_with_kv_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor | None,
) -> None:
    torch.ops._C.cpu_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        alibi_slopes,
        sliding_window[0],
        sliding_window[1],
        block_table,
        softcap,
        scheduler_metadata,
        s_aux,
    )


def cpu_gemm_wna16(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    bias: torch.Tensor | None,
    pack_factor: int,
    isa_hint: str,
) -> torch.Tensor:
    output = torch.empty((input.size(0), scales.size(1)), dtype=input.dtype)
    torch.ops._C.cpu_gemm_wna16(
        input,
        q_weight,
        output,
        scales,
        zeros,
        g_idx,
        bias,
        pack_factor,
        isa_hint,
    )
    return output


def cpu_prepack_moe_weight(
    weight: torch.Tensor,
    isa: str,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    torch.ops._C.prepack_moe_weight(weight, output, isa)
    return output


def cpu_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: str,
    isa: str,
) -> torch.Tensor:
    output = torch.empty_like(input)
    torch.ops._C.cpu_fused_moe(
        output,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        act,
        isa,
    )
    return output


if hasattr(torch.ops._qutlass_C, "matmul_mxf4_bf16_tn"):

    @register_fake("_qutlass_C::matmul_mxf4_bf16_tn")
    def _fake_matmul_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
    ):
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


def matmul_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._qutlass_C.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


if hasattr(torch.ops._qutlass_C, "matmul_ada_mxf4_bf16_tn"):

    @register_fake("_qutlass_C::matmul_ada_mxf4_bf16_tn")
    def _fake_matmul_ada_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
    ):
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


def matmul_ada_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._qutlass_C.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


def ceil_div(a, b):
    return (a + b - 1) // b


if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxQuest"):

    @register_fake("_qutlass_C::fusedQuantizeMxQuest")
    def _fake_fused_quantize_mx_quest(
        a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        return xh_e2m1, xh_e8m0


if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxAbsMax"):

    @register_fake("_qutlass_C::fusedQuantizeMxAbsMax")
    def _fake_fused_quantize_mx_absmax(
        a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        return xh_e2m1, xh_e8m0


def fusedQuantizeMx(
    a: torch.Tensor, b: torch.Tensor, *, method: Literal["quest", "abs_max"] = "quest"
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() == 0:
        raise ValueError("`a` must have at least 1 dimension.")
    if a.size(-1) % 32 != 0:
        raise ValueError(f"last dim of `a` must be divisible by 32, got {a.size(-1)}.")
    if b.device != a.device:
        raise ValueError("`a` and `b` must be on the same device.")

    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    rows, cols = a.numel() // a.size(-1), a.size(-1) // 32
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=a.device
    )

    if not hasattr(torch.ops, "_qutlass_C"):
        raise RuntimeError(
            "The `_qutlass_C` extension is not loaded. "
            "Make sure your custom op library is imported before calling fusedQuantizeMx."
        )

    if method == "quest":
        return torch.ops._qutlass_C.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
    elif method == "abs_max":
        return torch.ops._qutlass_C.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


if hasattr(torch.ops._qutlass_C, "fusedQuantizeNv"):

    @register_fake("_qutlass_C::fusedQuantizeNv")
    def _fake_fused_quantize_nv(
        a: torch.Tensor,
        b: torch.Tensor,
        xh_e2m1: torch.Tensor,
        xh_e4m3: torch.Tensor,
        global_scale: torch.Tensor,
    ):
        return xh_e2m1, xh_e4m3


def fusedQuantizeNv(
    a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    rows, cols = a.numel() // a.size(-1), a.size(-1) // 16
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device
    )

    return torch.ops._qutlass_C.fusedQuantizeNv(a, b, xh_e2m1, xh_e4m3, global_scale)


def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)
    kernels. Note that these kernels exploit the recursive properties of
    Sylvester Hadamards, and therefore do not require transform weight data

    Note that sylvester hadamard transforms are also symmetric, which means that
    this function is also applies the (transpose <=> inverse) transform.

    :param x: value to be transformed inplace
    :param inplace: modify value in place
    :return: value after transformation
    """
    return torch.ops._C.hadacore_transform(x, inplace)


if hasattr(torch.ops._C, "hadacore_transform"):

    @register_fake("_C::hadacore_transform")
    def _hadacore_transform_fake(x: torch.Tensor, inplace: bool) -> torch.Tensor:
        return torch.empty_like(x) if not inplace else x
