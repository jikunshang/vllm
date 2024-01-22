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
void reshape_and_cache_xpu_impl_(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int key_cache_stride,
    const int key_cache_num_tokens) {
  const int block_elem_num = num_heads * head_size * block_size;
  sycl::queue& q = vllm::xpu::vllmGetQueue();
  sycl::buffer<scalar_sycl_t> key_buf(
      (scalar_sycl_t*)key, num_tokens * key_stride);
  sycl::buffer<scalar_sycl_t> value_buf(
      (scalar_sycl_t*)value, num_tokens * value_stride);
  sycl::buffer<int64_t> slot_mapping_buf(slot_mapping, num_tokens);

  sycl::buffer<scalar_sycl_t> key_cache_buf(
      (scalar_sycl_t*)key_cache, key_cache_stride * key_cache_num_tokens);
  sycl::buffer<scalar_sycl_t> value_cache_buf(
      (scalar_sycl_t*)value_cache, key_cache_stride * key_cache_num_tokens);

  q.submit([&](auto& h) {
    sycl::accessor key_acc(key_buf, h, sycl::read_only);
    sycl::accessor value_acc(value_buf, h, sycl::read_only);
    sycl::accessor slot_mapping_acc(slot_mapping_buf, h, sycl::read_only);

    sycl::accessor key_cache_acc(key_cache_buf, h, sycl::read_write);
    sycl::accessor value_cache_acc(value_cache_buf, h, sycl::read_write);

    h.parallel_for(sycl::range(num_tokens, num_heads), [=](sycl::item<2> item) {
      size_t token_idx = item[0];
      size_t head_idx = item[1];
      const int64_t slot_idx = slot_mapping_acc[token_idx];
      if (slot_idx >= 0) {
        size_t src_key_head_idx = token_idx * key_stride + head_idx * head_size;
        size_t src_value_head_idx =
            token_idx * value_stride + head_idx * head_size;

        const int64_t block_index = slot_idx / block_size;
        const int64_t block_offset = slot_idx % block_size;

        for (int src_key_idx = 0; src_key_idx < head_size; src_key_idx += x) {
          const int64_t target_offset =
              src_key_idx * block_size + block_offset * x;
          for (int i = 0; i < x; ++i) {
            key_cache_acc
                [target_offset + i + block_elem_num * block_index +
                 head_idx * block_size * head_size] =
                    key_acc[src_key_idx + i + src_key_head_idx];
          }
        }

        for (int src_value_idx = 0; src_value_idx < head_size;
             ++src_value_idx) {
          const int64_t target_offset =
              src_value_idx * block_size + block_offset;
          value_cache_acc
              [target_offset + block_elem_num * block_index +
               head_idx * block_size * head_size] =
                  value_acc[src_value_idx + src_value_head_idx];
        }
      }
    });
  });
  q.wait();
}

template <typename scalar_t>
void reshape_and_cache_xpu_impl(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int key_cache_stride,
    const int key_cache_num_tokens) {
  reshape_and_cache_xpu_impl_<scalar_t, scalar_t>(
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      num_tokens,
      key_stride,
      value_stride,
      num_heads,
      head_size,
      block_size,
      x,
      key_cache_stride,
      key_cache_num_tokens);
}

template <>
void reshape_and_cache_xpu_impl<c10::Half>(
    const c10::Half* __restrict__ key,
    const c10::Half* __restrict__ value,
    c10::Half* __restrict__ key_cache,
    c10::Half* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int key_cache_stride,
    const int key_cache_num_tokens) {
  reshape_and_cache_xpu_impl_<c10::Half, sycl::half>(
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      num_tokens,
      key_stride,
      value_stride,
      num_heads,
      head_size,
      block_size,
      x,
      key_cache_stride,
      key_cache_num_tokens);
}

template <>
void reshape_and_cache_xpu_impl<c10::BFloat16>(
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    c10::BFloat16* __restrict__ key_cache,
    c10::BFloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int key_cache_stride,
    const int key_cache_num_tokens) {
  reshape_and_cache_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      key,
      value,
      key_cache,
      value_cache,
      slot_mapping,
      num_tokens,
      key_stride,
      value_stride,
      num_heads,
      head_size,
      block_size,
      x,
      key_cache_stride,
      key_cache_num_tokens);
}

void reshape_and_cache_xpu(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  int key_cache_stride = key_cache.stride(0);
  int key_cache_num_tokens = key_cache.size(0);

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "reshape_and_cache_xpu_impl", [&] {
        reshape_and_cache_xpu_impl<scalar_t>(
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens,
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x,
            key_cache_stride,
            key_cache_num_tokens);
      });
}

template <typename scalar_t, typename scalar_sycl_t>
void copy_blocks_xpu_impl_(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::vector<std::pair<int64_t, int64_t>> mapping_pairs,
    const int element_num_per_block,
    const int layer_num) {
  const size_t pair_num = mapping_pairs.size();
  const size_t block_bytes = sizeof(scalar_t) * element_num_per_block;
  sycl::queue& q = vllm::xpu::vllmGetQueue();

  for (int layer = 0; layer < layer_num; ++layer) {
    for (size_t pair = 0; pair < pair_num; ++pair) {
      int64_t source_offset = element_num_per_block * mapping_pairs[pair].first;
      int64_t target_offset =
          element_num_per_block * mapping_pairs[pair].second;
      scalar_sycl_t* key_cache_ptr =
          (scalar_sycl_t*)key_caches[layer].data_ptr<scalar_t>();
      scalar_sycl_t* source_ptr = key_cache_ptr + source_offset;
      scalar_sycl_t* target_ptr = key_cache_ptr + target_offset;
      q.memcpy(target_ptr, source_ptr, block_bytes);

      scalar_sycl_t* value_cache_ptr =
          (scalar_sycl_t*)value_caches[layer].data_ptr<scalar_t>();
      source_ptr = value_cache_ptr + source_offset;
      target_ptr = value_cache_ptr + target_offset;
      q.memcpy(target_ptr, source_ptr, block_bytes);
    }
  }
}

template <typename scalar_t>
void copy_blocks_xpu_impl(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::vector<std::pair<int64_t, int64_t>> mapping_pairs,
    const int element_num_per_block,
    const int layer_num) {
  copy_blocks_xpu_impl_<scalar_t, scalar_t>(
      key_caches,
      value_caches,
      mapping_pairs,
      element_num_per_block,
      layer_num);
}

template <>
void copy_blocks_xpu_impl<c10::Half>(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::vector<std::pair<int64_t, int64_t>> mapping_pairs,
    const int element_num_per_block,
    const int layer_num) {
  copy_blocks_xpu_impl_<c10::Half, sycl::half>(
      key_caches,
      value_caches,
      mapping_pairs,
      element_num_per_block,
      layer_num);
}

template <>
void copy_blocks_xpu_impl<typename c10::BFloat16>(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::vector<std::pair<int64_t, int64_t>> mapping_pairs,
    const int element_num_per_block,
    const int layer_num) {
  copy_blocks_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      key_caches,
      value_caches,
      mapping_pairs,
      element_num_per_block,
      layer_num);
}

void copy_blocks_xpu(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }

  std::vector<std::pair<int64_t, int64_t>> mapping_pairs;
  mapping_pairs.reserve(block_mapping.size());
  for (const auto& pair : block_mapping) {
    for (const auto& dst : pair.second) {
      mapping_pairs.emplace_back(pair.first, dst);
    }
  }

  const int element_num_per_block = key_caches[0][0].numel();
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key_caches[0].scalar_type(), "copy_blocks_xpu_impl", [&] {
        copy_blocks_xpu_impl<scalar_t>(
            key_caches,
            value_caches,
            mapping_pairs,
            element_num_per_block,
            num_layers);
      });
}

void swap_blocks_xpu(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::map<int64_t, int64_t>& block_mapping) {
  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());
  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  sycl::queue& q = vllm::xpu::vllmGetQueue();
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    q.memcpy(dst_ptr + dst_offset, src_ptr + src_offset, block_size_in_bytes);
  }
}

void gather_cached_kv_xpu(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {}