# ==------------------------------------------------------------------== #
# vLLM-HPU-EXTENSION PATCH 
# FIXME: (Yi) move it to vllm-hpu-extension before merging to the habana_main.
# ==------------------------------------------------------------------== #
import torch
import vllm_hpu_extension.ops as ops

def initialize_fp8_kv_cache(mod, load_device="hpu"):

    class FP8VLLMKVCache(torch.nn.Module):

        def __init__(self, mod):
            super().__init__()
            self.orig_mod = mod
            self.scale_input = torch.tensor(1.0,
                                            dtype=torch.bfloat16,
                                            device=load_device)

        def quant_input(self, x, input_scale=None):
            return torch.ops.hpu.cast_to_fp8_v2(x, input_scale, False, False,
                                                torch.float8_e4m3fn)[0]

        def forward(self, input, *args, **kwargs):
            qinput = self.quant_input(input, input_scale=self.scale_input)
            return self.orig_mod(qinput, *args, **kwargs)

        def fetch_from_cache(self, quant_cache, blocks, permutations=None):
            if permutations:
                output_cache = self.orig_mod.fetch_from_cache(
                    quant_cache, blocks, permutations)
                return output_cache
            output_cache = self.orig_mod.fetch_from_cache(quant_cache, blocks)
            return output_cache

    return FP8VLLMKVCache(mod)


def initialize_fp8_matmul(mod, load_device="hpu"):

    class FP8Matmul(torch.nn.Module):

        def __init__(self, mod):
            super().__init__()
            self.orig_mod = mod
            self.scale_input = torch.tensor(1.0,
                                            dtype=torch.bfloat16,
                                            device=load_device)
            self.scale_other = torch.tensor(1.0,
                                            dtype=torch.bfloat16,
                                            device=load_device)

        def quant_input(self, x):
            return torch.ops.hpu.cast_to_fp8_v2(x, 1.0, False, False,
                                                torch.float8_e4m3fn)[0]

        def matmul_fp8(self,
                       x,
                       other,
                       out_dtype,
                       scale_input_inv=None,
                       scale_other_inv=None):
            return torch.ops.hpu.fp8_gemm_v2(A=x,
                                             trans_A=False,
                                             B=other,
                                             trans_B=False,
                                             D=None,
                                             out_dtype=out_dtype,
                                             A_scale_inv=scale_input_inv,
                                             B_scale_inv=scale_other_inv,
                                             bias=None,
                                             accumulate=False)

        def forward(self, input, other):
            qinput = self.quant_input(input)
            qother = other
            #qother = self.quant_input_1(other)
            output = self.matmul_fp8(qinput,
                                     qother,
                                     out_dtype=torch.bfloat16,
                                     scale_input_inv=self.scale_input,
                                     scale_other_inv=self.scale_other)
            return output

    return FP8Matmul(mod)

def _pipelined_pa(attn, value, block_groups, block_mapping, block_scales,
                  batch_size, matmul_av_op, batch2block_matmul_op,
                  block2batch_matmul_op):
    # When fp32_softmax is enabled attn is left in fp32 after Q@K
    # We can return to native dtype after we renormalize and calculate
    # the adjustments

    # Normalize the attention scores and cast attn to native dtype
    block_max = attn.amax(dim=-1, keepdim=True)
    adjustment_target_shape = block_max.shape
    attn = attn.sub(block_max)
    attn = attn.exp()
    #attn = attn.to(value.dtype)
    block_sums = attn.sum(dim=-1, keepdim=True)
    attn = matmul_av_op(attn, value)
    # qother = self.quant_input_1(other)
    block_max = block_max.squeeze()
    block_sums = block_sums.squeeze()

    # Calculate maximum of blocks that belong to the same sequences
    # and cast adjustments to native dtype
    group_max = ops.grouped_max(block_max, batch_size, block_groups)
    block_adjustment = (block_max - group_max).exp()
    #block_adjustment = block_adjustment.to(value.dtype)
    sum_adjusted = block_sums.mul(block_adjustment)

    # Sum block's sums that belongs to the same sequences
    group_sum_adjusted = ops.block2batch(sum_adjusted, block_mapping,
                                         block2batch_matmul_op)
    group_sum_adjusted = ops.batch2block(group_sum_adjusted, block_mapping,
                                         batch2block_matmul_op)
    sum_adjusted = sum_adjusted.view(*adjustment_target_shape)
    group_sum_adjusted = group_sum_adjusted.view(*adjustment_target_shape)
    block_adjustment = block_adjustment.view(*adjustment_target_shape)

    # For stability in case some of the sums have been zeroed out during
    # block aggretation
    group_sum_adjusted = torch.maximum(group_sum_adjusted, sum_adjusted)

    # Post processing for the attention scores
    rescale = block_adjustment.div(group_sum_adjusted)
    attn = attn.mul(rescale)
    return attn

