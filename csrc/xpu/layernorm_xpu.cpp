// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on

#include <torch/extension.h>
#include <algorithm>
#include "xpu_types.hpp"
#include "utils.h"

template <typename scalar_t, typename scalar_sycl_t>
void rms_norm_xpu_impl_(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const size_t hidden_size) {
    sycl::queue& q = vllm::xpu::vllmGetQueue();
  sycl::buffer<scalar_sycl_t, 2> input_buf(
      (scalar_sycl_t*)input, sycl::range(num_tokens, hidden_size));
  sycl::buffer<scalar_sycl_t, 1> weight_buf(
      (scalar_sycl_t*)weight, hidden_size);
  sycl::buffer<scalar_sycl_t, 2> out_buf(
      (scalar_sycl_t*)out, sycl::range(num_tokens, hidden_size));

  size_t size_1 = 1;
  size_t size_2 = 64ul;
  sycl::range<2> global_size = sycl::range<2>{num_tokens, size_2};
  sycl::range<2> local_size = sycl::range<2>{size_1, size_2};
  scalar_sycl_t* accum_data =
      sycl::malloc_device<scalar_sycl_t>(num_tokens * size_2, q);

  sycl::buffer<scalar_sycl_t, 2> accum_buf(
      accum_data, sycl::range<2>{num_tokens, size_2});

  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(
        sycl::nd_range<2>(global_size, local_size),
        [=](sycl::nd_item<2> index) {
          size_t g_row_id = index.get_global_id()[0];
          size_t l_col_id = index.get_local_id()[1];
          int group_col_id = index.get_group(1);
          scalar_sycl_t sum = 0;
          for (int i = l_col_id; i < hidden_size; i += size_2) {
            scalar_sycl_t x = input_acc[g_row_id][i];
            sum += x * x;
          }
          accum_acc[g_row_id][l_col_id] = sum;
        });
  });
  q.wait();

  q.submit([&](auto& h) {
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(num_tokens, [=](auto index) {
      size_t row_id = index[0];
      for (int i = 1; i < size_2; ++i) {
        accum_acc[row_id][0] += accum_acc[row_id][i];
      }
      float tmp = (float)accum_acc[row_id][0] / (float)hidden_size + epsilon;
      accum_acc[row_id][0] = (scalar_sycl_t)sycl::rsqrt(tmp);
    });
  });
  q.wait();
  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor weight_acc(weight_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);
    sycl::accessor out_acc(out_buf, h, sycl::read_write);

    h.parallel_for(
        sycl::range<2>(num_tokens, hidden_size), [=](sycl::item<2> index) {
          size_t row_id = index[0];
          size_t col_id = index[1];
          scalar_sycl_t x = input_acc[row_id][col_id];
          out_acc[row_id][col_id] =
              ((scalar_sycl_t)(x * accum_acc[row_id][0])) * weight_acc[col_id];
        });
  });
  q.wait();
  sycl::free(accum_data, q);
  // free(accum_data);
  // q.wait();
}

template <typename scalar_t>
void rms_norm_xpu_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const size_t hidden_size) {
  rms_norm_xpu_impl_<scalar_t, scalar_t>(
      out, input, weight, epsilon, num_tokens, hidden_size);
}

template <>
void rms_norm_xpu_impl<typename c10::Half>(
    c10::Half* __restrict__ out,
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const size_t hidden_size) {
  rms_norm_xpu_impl_<c10::Half, sycl::half>(
      out, input, weight, epsilon, num_tokens, hidden_size);
}

template <>
void rms_norm_xpu_impl<typename c10::BFloat16>(
    c10::BFloat16* __restrict__ out,
    const c10::BFloat16* __restrict__ input,
    const c10::BFloat16* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const size_t hidden_size) {
  rms_norm_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      out, input, weight, epsilon, num_tokens, hidden_size);
}

void _rms_norm_xpu(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_xpu_impl", [&] {
        rms_norm_xpu_impl(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            epsilon,
            num_tokens,
            hidden_size);
      });
}

void rms_norm_xpu(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  switch (weight.device().type()) {
    case c10::DeviceType::XPU:
      return _rms_norm_xpu(out, input, weight, epsilon);
    default:
      TORCH_CHECK(false, "Unsupported device type.")
  }
}

template <typename scalar_t, typename scalar_sycl_t>
void fused_add_rms_norm_xpu_impl_(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const int hidden_size) {
    sycl::queue& q = vllm::xpu::vllmGetQueue();

  sycl::buffer<scalar_sycl_t, 2> input_buf(
      (scalar_sycl_t*)input, sycl::range(num_tokens, hidden_size));
  sycl::buffer<scalar_sycl_t, 2> residual_buf(
      (scalar_sycl_t*)residual, sycl::range(num_tokens, hidden_size));
  sycl::buffer<scalar_sycl_t, 1> weight_buf(
      (scalar_sycl_t*)weight, hidden_size);

  size_t size_1 = 1;
  size_t size_2 = 64ul;
  sycl::range<2> global_size = sycl::range<2>{num_tokens, size_2};
  sycl::range<2> local_size = sycl::range<2>{size_1, size_2};
  scalar_sycl_t* accum_data =
      sycl::malloc_device<scalar_sycl_t>(num_tokens * size_2, q);

  sycl::buffer<scalar_sycl_t, 2> accum_buf(
      accum_data, sycl::range<2>{num_tokens, size_2});

  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_write);
    sycl::accessor residual_acc(residual_buf, h, sycl::read_write);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(
        sycl::nd_range<2>(global_size, local_size),
        [=](sycl::nd_item<2> index) {
          size_t g_row_id = index.get_global_id()[0];
          size_t l_col_id = index.get_local_id()[1];
          int group_col_id = index.get_group(1);
          scalar_sycl_t sum = 0;
          for (int i = l_col_id; i < hidden_size; i += size_2) {
            input_acc[g_row_id][i] += residual_acc[g_row_id][i];
            scalar_sycl_t x = input_acc[g_row_id][i];
            residual_acc[g_row_id][i] = x;
            sum += x * x;
          }
          accum_acc[g_row_id][l_col_id] = sum;
        });
  });
  q.wait();

  q.submit([&](auto& h) {
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(num_tokens, [=](auto index) {
      size_t row_id = index[0];
      for (int i = 1; i < size_2; ++i) {
        accum_acc[row_id][0] += accum_acc[row_id][i];
      }
      float tmp = (float)accum_acc[row_id][0] / (float)hidden_size + epsilon;
      accum_acc[row_id][0] = (scalar_sycl_t)sycl::rsqrt(tmp);
    });
  });
  q.wait();
  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_write);
    sycl::accessor weight_acc(weight_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(
        sycl::range<2>(num_tokens, hidden_size), [=](sycl::item<2> index) {
          size_t row_id = index[0];
          size_t col_id = index[1];
          scalar_sycl_t x = input_acc[row_id][col_id];
          input_acc[row_id][col_id] =
              ((scalar_sycl_t)(x * accum_acc[row_id][0])) * weight_acc[col_id];
        });
  });
  q.wait();
  sycl::free(accum_data, q);
}

template <typename scalar_t>
void fused_add_rms_norm_xpu_impl(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const int hidden_size) {
  fused_add_rms_norm_xpu_impl_<scalar_t, scalar_t>(
      input, residual, weight, epsilon, num_tokens, hidden_size);
}

template <>
void fused_add_rms_norm_xpu_impl<typename c10::Half>(
    c10::Half* __restrict__ input,
    c10::Half* __restrict__ residual,
    const c10::Half* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const int hidden_size) {
  fused_add_rms_norm_xpu_impl_<c10::Half, sycl::half>(
      input, residual, weight, epsilon, num_tokens, hidden_size);
}

template <>
void fused_add_rms_norm_xpu_impl<typename c10::BFloat16>(
    c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ residual,
    const c10::BFloat16* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const int hidden_size) {
  fused_add_rms_norm_xpu_impl_<c10::BFloat16, sycl::ext::oneapi::bfloat16>(
      input, residual, weight, epsilon, num_tokens, hidden_size);
}

void _fused_add_rms_norm_xpu(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_xpu_impl", [&] {
        fused_add_rms_norm_xpu_impl(
            input.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            epsilon,
            num_tokens,
            hidden_size);
      });
}

void fused_add_rms_norm_xpu(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  switch (weight.device().type()) {
    case c10::DeviceType::XPU:
      return _fused_add_rms_norm_xpu(input, residual, weight, epsilon);
    default:
      TORCH_CHECK(false, "Unsupported device type.")
  }
}