#pragma once

#include <sycl/sycl.hpp>
#include <memory>

namespace vllm {
namespace xpu {

static std::unique_ptr<sycl::queue> g_queue;

static bool queue_init = false;

static void initGlobalQueue() {
  g_queue = std::make_unique<sycl::queue>(sycl::queue(sycl::gpu_selector_v));
}

static inline sycl::queue& vllmGetQueue() {
  if (!queue_init) {
    initGlobalQueue();

    queue_init = true;

  } else {
    return *g_queue;
  }
}

} // namespace xpu

} // namespace vllm
