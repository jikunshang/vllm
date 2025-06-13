# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


@dataclass
class IPEXAttentionMetadata(FlashAttentionMetadata):
    seq_start_loc: torch.Tensor = torch.tensor([0], dtype=torch.int64)
    decode_num: int = 0
    have_prompt: bool = False


class IPEXAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "IPEX_V1"

    @staticmethod
    def get_impl_cls() -> type["IPEXAttentionImpl"]:
        return IPEXAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return IPEXAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, num_kv_heads, head_size, block_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        # TODO: support cascade attention
        return False


class IPEXAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = IPEXAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IpexAttnBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IPEXAttentionBackend,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEXAttention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output
        # print(attn_metadata.decode_num)
        decode_num = attn_metadata.decode_num

        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)
        # Reshape the input keys and values and store them in the cache.
        key_cache, value_cache = kv_cache.unbind(0)
        (num_blocks, num_kv_heads, head_size, block_size) = key_cache.shape

        # 0. write kv to cache.
        ipex_ops.reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache.view(num_blocks, num_kv_heads, head_size,
                                     block_size, 1),
            value_cache=value_cache,
            slot_mapping=attn_metadata.slot_mapping.flatten(),
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
        )

        # 1. process decode if any
        if decode_num > 0:
            ipex_ops.paged_attention_v1(
                out=output[:decode_num],
                query=query[:decode_num],
                key_cache=key_cache.view(num_blocks, num_kv_heads, head_size,block_size, 1),
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                scale=self.scale,
                block_tables=attn_metadata.block_table,
                context_lens=attn_metadata.seq_lens[:decode_num],
                block_size=block_size,
                max_context_len=attn_metadata.max_seq_len, 
                alibi_slopes=self.alibi_slopes,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
            )
        # 2. process prefill if any
        if attn_metadata.have_prompt:

            ipex_ops.varlen_attention(
                query=query[decode_num:,],
                key=key[decode_num:,],
                value=value[decode_num:,],
                out=output[decode_num:,],
                seqlen_q=attn_metadata.seq_start_loc[decode_num:,],
                seqlen_k=attn_metadata.seq_start_loc[decode_num:,],
                alibi_slopes=self.alibi_slopes,
                max_seqlen_q=attn_metadata.max_seq_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                pdropout=0.0,
                softmax_scale=self.scale,
                zero_tensors=False,
                is_causal=True,
                return_softmax=False,
                gen_=None,
                window_size_left=-1,
                window_size_right=-1,
                logits_soft_cap=self.logits_soft_cap,
            )

        return

    def forward_chunk_prefill(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IPEXAttentionBackend,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEXAttention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.random_(0, 10)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        # Reshape the input keys and values and store them in the cache.
        key_cache, value_cache = kv_cache.unbind(0)

        ipex_ops.reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale_float,
            layer._v_scale_float,
        )

        ipex_ops.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            None,
            attn_metadata.block_table,
            self.alibi_slopes,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            0.0,
            self.scale,
            False,
            self.sliding_window[0],
            self.sliding_window[1],
            True,
            False,
            None,
            self.kv_cache_dtype,
        )
        return output
