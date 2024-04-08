import torch
import intel_extension_for_pytorch as ipex
from typing import Dict, Optional


class xpu_ops:

    @staticmethod
    def reshape_activation_tensor(x: torch.Tensor):
        num = x.size(0)
        d = x.size(1) // 2
        x = x.reshape(num, 2, d)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = x1.reshape(num, d)
        x2 = x2.reshape(num, d)
        return x1, x2

    def silu_and_mul(out: torch.Tensor, x: torch.Tensor):
        x1, x2 = xpu_ops.reshape_activation_tensor(x)
        torch.ops.torch_ipex.silu_mul(x1, x2, out)

    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor):
        x1, x2 = xpu_ops.reshape_activation_tensor(x)
        torch.ops.torch_ipex.gelu_mul(x1, x2, out, "none")

    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor):
        x1, x2 = xpu_ops.reshape_activation_tensor(x)
        torch.ops.torch_ipex.gelu_mul(x1, x2, out, "none")

    def gelu_fast(out: torch.Tensor, x: torch.Tensor):
        out.copy_(torch.nn.functional.gelu(x))

    def gelu_new(out: torch.Tensor, x: torch.Tensor):
        out.copy_(torch.nn.functional.gelu(x))

    def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        kv_scale: float,
    ):
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        head_mapping = torch.arange(
            0,
            num_kv_heads,
            device="xpu",
            dtype=torch.int32,
        ).view(num_kv_heads,
               1).repeat_interleave(num_queries_per_tokens).flatten()
        ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
            out, query, key_cache, value_cache, head_mapping, scale,
            block_tables, context_lens, block_size, max_context_len,
            alibi_slopes)

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
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        kv_scale: float,
    ):
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        head_mapping = torch.arange(
            0,
            num_kv_heads,
            dtype=torch.int32,
        ).view(num_kv_heads,
               1).repeat_interleave(num_queries_per_tokens).flatten()
        torch.xpu.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query,
                                     key_cache, value_cache, head_mapping,
                                     block_tables, context_lens, scale,
                                     block_size, max_context_len, alibi_slopes)

    def rotary_embedding(
        positions: torch.Tensor,  # [batch_size, seq_len]
        query: torch.Tensor,  # [batch_size, seq_len, num_heads*head_size]
        key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads*head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [cos_sin_dim, rot_dim]
        is_neox: bool,
    ):
        rotary_dim = cos_sin_cache.size(1)
        query = query.view(*query.shape[:-1], -1, head_size)
        key = key.view(*key.shape[:-1], -1, head_size)

        query_rot = query[..., :rotary_dim]
        key_rot = key[..., :rotary_dim]
        if rotary_dim < head_size:
            query_pass = query[..., rotary_dim:]
            key_pass = key[..., rotary_dim:]

        cos_sin = cos_sin_cache[positions.long()]
        cos, sin = cos_sin.chunk(2, dim=-1)

        if is_neox:
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            torch.ops.torch_ipex.apply_rotary_embedding_half_qk(
                query_rot, key_rot, sin, cos, query_rot, key_rot)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
            torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
                query_rot, key_rot, sin, cos, query_rot, key_rot)

    def batched_rotary_embedding(positions: torch.tensor, query: torch.tensor,
                                 key: torch.tensor, head_size: int,
                                 cos_sin_cache: torch.tensor, is_neox: bool,
                                 rot_dim: int,
                                 cos_sin_cache_offsets: torch.tensor):
        raise RuntimeError(
            "Not supported op: batched_rotary_embedding for xpu backend")

    def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float):
        tmp = ipex.llm.modules.RMSNorm.apply(input, weight, epsilon)
        out.copy_(tmp)

    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float):
        tmp = ipex.llm.functional.add_rms_norm(residual, input, weight, None,
                                               epsilon, True)
        input.copy_(tmp)


class xpu_cache_ops:

    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        kv_scale: float,
    ) -> None:
        assert kv_cache_dtype == "auto"
        torch.ops.torch_ipex.reshape_and_cache(key, value, key_cache,
                                               value_cache, slot_mapping)

    def copy_blocks(key_caches, value_caches, block_mapping):
        block_mapping_tensor = []
        for key, values in block_mapping.items():
            if hasattr(values, "__iter__"):
                for value in values:
                    block_mapping_tensor.append([key, value])
        block_mapping = torch.tensor(block_mapping_tensor,
                                     device="xpu",
                                     dtype=torch.int64)
        torch.ops.torch_ipex.copy_blocks(key_caches, value_caches,
                                         block_mapping_tensor)

    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping: Dict[int, int]):
        keys = list(block_mapping.keys())
        values = list(block_mapping.values())
        key_tensor = torch.tensor(keys)
        value_tensor = torch.tensor(values)
        block_mapping_tensor = torch.stack([key_tensor, value_tensor], dim=1)

        torch.ops.torch_ipex.swap_blocks(src, dst, block_mapping_tensor)
