#pragma once

#include <sycl/sycl.hpp>
#include <memory>
// #include <ipex.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#define VLLM_LDG(arg) *(arg)
namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue() {
  auto current_stream = c10::xpu::getCurrentXPUStream();
  auto& queue = current_stream.queue();
  return queue;
}
template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

}  // namespace xpu

}  // namespace vllm

SYCL_EXTERNAL sycl::half sycl_half_mul(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_add(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_sub(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_fma(sycl::half a, sycl::half b,
                                       sycl::half c);

SYCL_EXTERNAL sycl::half2 sycl_half_mul2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_add2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_sub2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_fma2(sycl::half2 a, sycl::half2 b,
                                         sycl::half2 c);

int get_max_shared_memory_per_block_device_attribute(int device_id);