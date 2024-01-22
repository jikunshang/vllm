// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on
#include "xpu_types.hpp"

#include <torch/extension.h>
#include "utils.h"

template <typename T>
__inline__ T silu_xpu(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename scalar_t, typename scalar_sycl_t>
void silu_and_mul_xpu_impl_(
    int num_tokens,
    int d,
    const scalar_t* __restrict__ input, //
    scalar_t* __restrict__ output) {
  sycl::queue& q = vllm::xpu::vllmGetQueue();
  sycl::buffer<scalar_sycl_t, 1> input_buf(
      (scalar_sycl_t*)input, num_tokens * d * 2);
  sycl::buffer<scalar_sycl_t, 1> output_buf(
      (scalar_sycl_t*)output, num_tokens * d);
  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor output_acc(output_buf, h, sycl::read_write);

    // each work item calculate 16 output result, trying to leverage SIMD lane
    h.parallel_for(sycl::range<1>(num_tokens * d), [=](sycl::item<1> index) {
      int i = index[0];
      int token_idx = i / d;
      int dim_idx = i % d;
      const scalar_sycl_t x = input_acc[token_idx * d * 2 + dim_idx];
      const scalar_sycl_t y = input_acc[token_idx * d * 2 + dim_idx + d];
      output_acc[token_idx * d + dim_idx] = silu_xpu(x) * y;
    });
  });
  q.wait();
}

template <typename scalar_t>
void silu_and_mul_xpu_impl(
    int num_tokens,
    int d,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {
  silu_and_mul_xpu_impl_<scalar_t, scalar_t>(num_tokens, d, input, output);
}

template <>
void silu_and_mul_xpu_impl<typename c10::Half>(
    int num_tokens,
    int d,
    const c10::Half* __restrict__ input,
    c10::Half* __restrict__ output) {
  silu_and_mul_xpu_impl_<c10::Half, sycl::half>(num_tokens, d, input, output);
}

template <>
void silu_and_mul_xpu_impl<typename c10::BFloat16>(
    int num_tokens,
    int d,
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output) {
  silu_and_mul_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      num_tokens, d, input, output);
}

void silu_and_mul_xpu(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_xpu_impl", [&] {
        silu_and_mul_xpu_impl(
            num_tokens,
            d,
            input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>());
      });
}

// void gelu_new(torch::Tensor &out, torch::Tensor &input);

// void gelu_fast(torch::Tensor &out, torch::Tensor &input);