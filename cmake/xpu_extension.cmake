set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# Define environment variables for special configurations
#
# TODO: detect Intel GPU Architecture(PVC or Arc) to add AOT flag.

#
# Check the compile flags
#
append_cmake_prefix_path("intel_extension_for_pytorch" "intel_extension_for_pytorch.cmake_prefix_path")
find_package(IPEX REQUIRED)
# IPEX will overwrite TORCH_LIBRARIES, so re-add torch_python lib.
append_torchlib_if_found(torch_python)
include_directories(${IPEX_INCLUDE_DIRS})
set(CMPLR_ROOT $ENV{CMPLR_ROOT})
set(CMAKE_CXX_COMPILER icpx)
set(VLLM_EXTRA_INCLUDE_DIRECTORIES ${CMPLR_ROOT}/include/sycl)

list(APPEND VLLM_GPU_FLAGS "-DVLLM_BUILD_XPU_OPS" "-fsycl" "-fsycl-targets=spir64")
list(APPEND VLLM_GPU_LINK_FLAGS "-fsycl" "-fsycl-targets=spir64")
list(APPEND VLLM_LINK_LIBRARIES "sycl" "OpenCL" "pthread" "m" "dl" "dnnl" "intel-ext-pt-gpu" )    

#
# Define extension targets
#

#
# _C extension
#
set(VLLM_EXT_SRC
    "csrc/xpu/activation_xpu.cpp"
    "csrc/xpu/attention_xpu.cpp"
    "csrc/xpu/cache_ops_xpu.cpp"
    "csrc/xpu/gemm_kernels_xpu.cpp"
    "csrc/xpu/layernorm_xpu.cpp"
    "csrc/xpu/pos_encoding_xpu.cpp"
    "csrc/xpu/utils.cpp"
    "csrc/pybind.cpp")

define_gpu_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${VLLM_EXTRA_INCLUDE_DIRECTORIES}
    LIBRARIES ${VLLM_LINK_LIBRARIES}
    WITH_SOABI
)

add_custom_target(default)
message(STATUS "Enabling C extension.")
add_dependencies(default _C)

