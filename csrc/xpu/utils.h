#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <ipex.h>
#include <ATen/ATen.h>

#define VLLM_LDG(arg) *(arg)
namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue() {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = ::xpu::get_queue_from_stream(c10_stream);
  return queue;
}
template <typename T>
struct SyclTypeTrait{
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half>{
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16>{
  using Type = sycl::ext::oneapi::bfloat16;
};


} // namespace xpu

} // namespace vllm
