// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on
#include "xpu_types.hpp"

#include <torch/extension.h>
#include "utils.h"

template <typename scalar_t, typename scalar_sycl_t>
void rotary_embedding_xpu_impl_(
    const int64_t* __restrict__ positions, // [num_tokens]
    scalar_t* __restrict__ query, // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache, // [max_position, 2, rot_dim //
                                                // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens,
    const int sin_cos_dim) {
  const int embed_dim = rot_dim / 2;
  sycl::buffer<int64_t, 1> positions_buf(positions, num_tokens);
  sycl::buffer<scalar_sycl_t, 3> query_buf(
      (scalar_sycl_t*)query, sycl::range<3>(num_tokens, num_heads, head_size));
  sycl::buffer<scalar_sycl_t, 3> key_buf(
      (scalar_sycl_t*)key, sycl::range<3>(num_tokens, num_kv_heads, head_size));
  sycl::buffer<scalar_sycl_t, 1> cos_sin_cache_buf(
      (scalar_sycl_t*)cos_sin_cache, sin_cos_dim * rot_dim);
    sycl::queue& q = vllm::xpu::vllmGetQueue();
  q.submit([&](auto& h) {
    sycl::accessor positions_acc(positions_buf, h, sycl::read_only);
    sycl::accessor query_acc(query_buf, h, sycl::read_write);
    sycl::accessor key_acc(key_buf, h, sycl::read_write);
    sycl::accessor cos_sin_cache_acc(cos_sin_cache_buf, h, sycl::read_only);

    h.parallel_for(
        sycl::range(num_tokens, num_heads, embed_dim), [=](auto index) {
          const long token_idx = index[0];
          const long head_idx = index[1];
          const int x_index = index[2];
          const int y_index = x_index + embed_dim;
          int64_t pos = positions_acc[token_idx];

          const scalar_sycl_t q_x = query_acc[token_idx][head_idx][x_index];
          const scalar_sycl_t q_y = query_acc[token_idx][head_idx][y_index];
          const scalar_sycl_t cos = cos_sin_cache_acc[pos * rot_dim + x_index];
          const scalar_sycl_t sin = cos_sin_cache_acc[pos * rot_dim + y_index];

          query_acc[token_idx][head_idx][x_index] = q_x * cos - q_y * sin;
          query_acc[token_idx][head_idx][y_index] = q_y * cos + q_x * sin;

          if (head_idx < num_kv_heads) {
            const scalar_sycl_t k_x = key_acc[token_idx][head_idx][x_index];
            const scalar_sycl_t k_y = key_acc[token_idx][head_idx][y_index];
            key_acc[token_idx][head_idx][x_index] = k_x * cos - k_y * sin;
            key_acc[token_idx][head_idx][y_index] = k_y * cos + k_x * sin;
          }
        });
  });

  q.wait();
}

template <typename scalar_t>
void rotary_embedding_xpu_impl(
    const int64_t* __restrict__ positions, // [num_tokens]
    scalar_t* __restrict__ query, // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache, // [max_position, 2, rot_dim //
                                                // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens,
    const int sin_cos_dim) {
  rotary_embedding_xpu_impl_<scalar_t, scalar_t>(
      positions,
      query,
      key,
      cos_sin_cache,
      rot_dim,
      query_stride,
      key_stride,
      num_heads,
      num_kv_heads,
      head_size,
      num_tokens,
      sin_cos_dim);
}

template <>
void rotary_embedding_xpu_impl<c10::Half>(
    const int64_t* __restrict__ positions, // [num_tokens]
    c10::Half* __restrict__ query, // [num_tokens, num_heads, head_size]
    c10::Half* __restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const c10::Half* __restrict__ cos_sin_cache, // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens,
    const int sin_cos_dim) {
  rotary_embedding_xpu_impl_<c10::Half, sycl::half>(
      positions,
      query,
      key,
      cos_sin_cache,
      rot_dim,
      query_stride,
      key_stride,
      num_heads,
      num_kv_heads,
      head_size,
      num_tokens,
      sin_cos_dim);
}

template <>
void rotary_embedding_xpu_impl<c10::BFloat16>(
    const int64_t* __restrict__ positions, // [num_tokens]
    c10::BFloat16* __restrict__ query, // [num_tokens, num_heads, head_size]
    c10::BFloat16* __restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const c10::BFloat16* __restrict__ cos_sin_cache, // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens,
    const int sin_cos_dim) {
  rotary_embedding_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      positions,
      query,
      key,
      cos_sin_cache,
      rot_dim,
      query_stride,
      key_stride,
      num_heads,
      num_kv_heads,
      head_size,
      num_tokens,
      sin_cos_dim);
}

void _rotary_embedding_xpu(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    torch::Tensor& cos_sin_cache) {
  // TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float);

  int num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int key_stride = key.stride(-2);
  int query_stride = query.stride(-2);
  int cos_sin_dim = cos_sin_cache.size(0);

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_xpu_impl", [&] {
        rotary_embedding_xpu_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rot_dim,
            query_stride,
            key_stride,
            num_heads,
            num_kv_heads,
            head_size,
            num_tokens,
            cos_sin_dim);
      });
}

void rotary_embedding_xpu(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  TORCH_CHECK(is_neox);
  switch (positions.device().type()) {
    case c10::DeviceType::XPU:
      return _rotary_embedding_xpu(
          positions, query, key, head_size, cos_sin_cache);
    default:
      TORCH_CHECK(false, "Unsupported device type.")
  }
}