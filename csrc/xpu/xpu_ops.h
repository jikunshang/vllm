#include <torch/extension.h>

void rotary_embedding_xpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox);

void silu_and_mul_xpu(torch::Tensor &out, torch::Tensor &input);

void gelu_new_xpu(torch::Tensor &out, torch::Tensor &input) {}

void gelu_fast_xpu(torch::Tensor &out, torch::Tensor &input) {}


void paged_attention_v1_xpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes);

void paged_attention_v2_xpu(
    torch::Tensor &out, torch::Tensor &exp_sums, torch::Tensor &max_logits,
    torch::Tensor &tmp_out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes);

void copy_blocks_xpu(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache_xpu(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping);

void swap_blocks_xpu(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping);

void gather_cached_kv_xpu(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping);

void rms_norm_xpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon);

void fused_add_rms_norm_xpu(torch::Tensor &input, torch::Tensor &residual,
                            torch::Tensor &weight, float epsilon);

inline torch::Tensor awq_gemm_xpu(torch::Tensor _in_feats, torch::Tensor _kernel,
                           torch::Tensor _scaling_factors, torch::Tensor _zeros,
                           int split_k_iters) {
  TORCH_CHECK(false, "Quantization is not supported on XPU.");
}

inline void squeezellm_gemm_xpu(torch::Tensor vec, torch::Tensor mat,
                         torch::Tensor mul, torch::Tensor lookup_table) {
  TORCH_CHECK(false, "Quantization is not supported on XPU.");
}

inline torch::Tensor gptq_gemm_xpu(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline void gptq_shuffle_xpu(
  torch::Tensor q_weight,
  torch::Tensor q_perm) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}