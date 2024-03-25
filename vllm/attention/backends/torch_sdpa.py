"""Attention layer with torch scaled_dot_product_attention and PagedAttention."""
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata


class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPAImpl"]:
        return TorchSDPAImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "TorchSDPAMetadata":
        return TorchSDPAMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TorchSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool



def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias

def _make_sliding_window_bias(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype, 
) -> torch.Tensor:
    tensor = torch.full(
        (1, seq_len, seq_len),
        dtype=dtype,
        fill_value=1,
    )
    shift = 0
    mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
    mask = torch.triu(mask, diagonal=shift - window_size + 1)
    mask = torch.log(mask)
    return mask.to(dtype)

class TorchSDPAImpl(AttentionImpl):
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.need_mask = (alibi_slopes is not None) or (sliding_window is not None)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: TorchSDPAMetadata,
    ) -> torch.Tensor:
        """Forward pass with torch scaled_dot_product_attention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size) 

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype)

        if attn_metadata.is_prompt:
            if kv_cache is None or attn_metadata.block_tables.numel() == 0:

                if self.num_kv_heads != self.num_heads:
                    key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
                    value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=1)

                attn_bias: torch.tensor = None
                if self.need_mask:
                    if self.alibi_slopes is not None:
                        att_bias = _make_alibi_bias(self.alibi_slopes, self.num_kv_heads, 1, num_tokens, query.dtype)
                    elif self.sliding_window is not None:
                        att_bias = _make_sliding_window_bias(num_tokens, self.sliding_window, query.dtype)
                    attn_bias = att_bias.to(query.device)

                # query = query.unflatten(0, (batch_size, seq_len)) 
                # key = key.unflatten(0, (batch_size, seq_len))
                # value = value.unflatten(0, (batch_size, seq_len))

                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
                print(query.shape)
                print(key.shape)
                print(value.shape)
                                        
                query = query.movedim(1, query.dim() - 2)
                key = key.movedim(1, key.dim() - 2)
                value = value.movedim(1, value.dim() - 2)
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, 
                    key, 
                    value, 
                    attn_bias,
                    0.0, 
                    is_causal=not self.need_mask,
                    scale=self.scale).movedim(query.dim() - 2, 1).contiguous()
                # output = out.view_as(query)
                # FIXME: half input will generate float output, next ipex release will fix this.
                output = out.to(query.dtype)
                
            else:
                # prefix-enabled attention
                raise RuntimeError("SDPA backend doesn't support prefix decoding.") 

        else:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )
        print(f"self.num_heads: {self.num_heads}, hidden_size: {hidden_size}, output shape: {output.shape}")
        # Reshape the output tensor.
        return output.view(-1 , self.num_heads * self.head_size)
