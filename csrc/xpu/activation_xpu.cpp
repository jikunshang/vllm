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

template <typename scalar_t>
void silu_and_mul_kernel(
    scalar_t* __restrict__ out, // [..., d]
    const scalar_t* __restrict__ input, // [..., 2, d]
    const int d,
    const sycl::nd_item<3>& item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    const scalar_t x = input[token_idx * 2 * d + idx];
    const scalar_t y = input[token_idx * 2 * d + d + idx];
    out[token_idx * d + idx] = silu_xpu(x) * y;
  }
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
void call_silu_and_mul_kernel(
    int num_tokens,
    int d,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(d, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          silu_and_mul_kernel<sycl_t>(
              (sycl_t*)output, (const sycl_t*)input, d, item_ct1);
        });
  });
}

void silu_and_mul_xpu(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_silu_and_mul_kernel", [&] {
        call_silu_and_mul_kernel(
            num_tokens,
            d,
            input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>());
      });
}

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
void activation_kernel(
    scalar_t* __restrict__ out, // [..., d]
    const scalar_t* __restrict__ input, // [..., d]
    const int d,
    const sycl::nd_item<3>& item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

template <typename T>
__inline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__inline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename scalar_t>
void call_gelu_new_activation_kernel(torch::Tensor& out, torch::Tensor& input) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(d, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          activation_kernel<sycl_t, gelu_new_kernel>(
              (sycl_t* __restrict__)out_ptr,
              (const sycl_t* __restrict__)input_ptr,
              d,
              item_ct1);
        });
  });
}

template <typename scalar_t>
void call_gelu_fast_activation_kernel(
    torch::Tensor& out,
    torch::Tensor& input) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(d, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          activation_kernel<sycl_t, gelu_fast_kernel>(
              (sycl_t* __restrict__)out_ptr,
              (const sycl_t* __restrict__)input_ptr,
              d,
              item_ct1);
        });
  });
}

void gelu_new_xpu(torch::Tensor& out, torch::Tensor& input) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "call_gelu_new_activation_kernel", [&] {
        call_gelu_new_activation_kernel<scalar_t>(out, input);
      });
}

void gelu_fast_xpu(torch::Tensor& out, torch::Tensor& input) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "call_gelu_fast_activation_kernel", [&] {
        call_gelu_fast_activation_kernel<scalar_t>(
            out, input);
      });
}