// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on
#include "xpu_types.hpp"

#include <torch/extension.h>
#include "utils.h"


template <typename scalar_t>
void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key, // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache, // [num_blocks, num_heads, head_size/x,
                                      // block_size, x]
    scalar_t* __restrict__ value_cache, // [num_blocks, num_heads, head_size,
                                        // block_size]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const sycl::nd_item<3>& item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range(2)) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

template <typename scalar_t>
void call_reshape_and_cache_kernel(
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
    const int x) {
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(num_heads * head_size, 512));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          reshape_and_cache_kernel<scalar_t>(
              key,
              value,
              key_cache,
              value_cache,
              slot_mapping,
              key_stride,
              value_stride,
              num_heads,
              head_size,
              block_size,
              x,
              item_ct1);
        });
  });
}

template <>
void call_reshape_and_cache_kernel<c10::Half>(
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
    const int x) {
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1,1,std::min(num_heads * head_size, 512));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          reshape_and_cache_kernel<sycl::half>(
              (sycl::half*)key,
              (sycl::half*)value,
              (sycl::half*)key_cache,
              (sycl::half*)value_cache,
              slot_mapping,
              key_stride,
              value_stride,
              num_heads,
              head_size,
              block_size,
              x,
              item_ct1);
        });
  });
}

template <>
void call_reshape_and_cache_kernel<c10::BFloat16>(
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
    const int x) {
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1,1,std::min(num_heads * head_size, 512));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          reshape_and_cache_kernel<sycl::ext::oneapi::bfloat16>(
              (sycl::ext::oneapi::bfloat16*)key,
              (sycl::ext::oneapi::bfloat16*)value,
              (sycl::ext::oneapi::bfloat16*)key_cache,
              (sycl::ext::oneapi::bfloat16*)value_cache,
              slot_mapping,
              key_stride,
              value_stride,
              num_heads,
              head_size,
              block_size,
              x,
              item_ct1);
        });
  });
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

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "call_reshape_and_cache_kernel", [&] {
        call_reshape_and_cache_kernel<scalar_t>(
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
            x);
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