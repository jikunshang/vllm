/*
 * Copyright (c) 2023, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <dpct/dpct.hpp>
#include <stdint.h>
#include <sycl/sycl.hpp>
typedef uint8_t fp8_sim;
namespace vllm {

inline float half_to_float(sycl::half h) {
  return float(h);
}

inline sycl::float2 half2_to_float2(sycl::half2 v) {
  return sycl::float2(half_to_float(v.x()), half_to_float(v.y()));
}

inline sycl::float4 half4_to_float4(sycl::half4 v) {
  return sycl::float4(half_to_float(v.x()), half_to_float(v.y()),
                      half_to_float(v.z()), half_to_float(v.w()));
}

inline sycl::half float_to_half(float f) {
  return sycl::half(f);
}

inline sycl::half2 float2_to_half2(sycl::float2 f) {
  return sycl::half2{float_to_half(f.x()), float_to_half(f.y())};
}


template <typename T_OUT, typename T_IN>
inline T_OUT convert(const T_IN* in) {
  T_OUT out = *(T_OUT*)(in);
  return out;
}

template <>
inline sycl::half convert<sycl::half, fp8_sim>(const fp8_sim* in) {
  sycl::half out{0};
  fp8_sim* out_p = (fp8_sim*)(&out);
  memcpy(out_p + 1, in, 1);
  return out;
}

template <>
inline fp8_sim convert<fp8_sim, sycl::half>(const sycl::half* in) {
  fp8_sim out{0};
  fp8_sim* in_p = (fp8_sim*)(in);
  memcpy(&out, in_p+1, 1);
  return out;
}

template <>
inline sycl::half2 convert<sycl::half2, fp8_sim>(const fp8_sim* in) {
  sycl::half2 out{0};
  fp8_sim* out_p = (fp8_sim*)(&out);
  memcpy(out_p + 1, in, 1);
  memcpy(out_p + 3, in+1, 1);
  return out;
}
template <>
inline sycl::half4 convert<sycl::half4, fp8_sim>(const fp8_sim* in) {
  sycl::half4 out{0};
  fp8_sim* out_p = (fp8_sim*)(&out);
  memcpy(out_p + 1, in, 1);
  memcpy(out_p + 3, in+1, 1);
  memcpy(out_p + 5, in+2, 1);
  memcpy(out_p + 7, in+3, 1);
  return out;
}
template <>
inline sycl::half8 convert<sycl::half8, fp8_sim>(const fp8_sim* in) {
  sycl::half8 out{0};
  fp8_sim* out_p = (fp8_sim*)(&out);
  memcpy(out_p + 1, in, 1);
  memcpy(out_p + 3, in+1, 1);
  memcpy(out_p + 5, in+2, 1);
  memcpy(out_p + 7, in+3, 1);
  memcpy(out_p + 9, in+4, 1);
  memcpy(out_p + 11, in+5, 1);
  memcpy(out_p + 13, in+6, 1);
  memcpy(out_p + 15, in+7, 1);  
  return out;
}

template <>
inline float convert<float, fp8_sim>(const fp8_sim* in) {
  // return 0;
  sycl::half h_out = convert<sycl::half, fp8_sim>(in);
  return half_to_float(h_out);
}

template <>
inline fp8_sim convert<fp8_sim, float>(const float* in) {
  sycl::half h = float_to_half(*in);
  return convert<fp8_sim, sycl::half>(&h);
}

template <>
inline sycl::float2 convert<sycl::float2, fp8_sim>(const fp8_sim* in) {
  sycl::half2 h_out = convert<sycl::half2, uint8_t>(in);
  return half2_to_float2(h_out);

}
template <>
inline sycl::float4 convert<sycl::float4, fp8_sim>(const fp8_sim* in) {
  sycl::half4 h_out = convert<sycl::half4, uint8_t>(in);
  return half4_to_float4(h_out);
}

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B>
inline Acc mul(A a, B b);

template <typename T>
inline float sum(T v);

template <typename T>
inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template <typename T>
inline void zero(T& dst) {
  constexpr int WORDS = (sizeof(T) / 4) == 0 ? 1 : (sizeof(T) / 4);
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

} // namespace vllm