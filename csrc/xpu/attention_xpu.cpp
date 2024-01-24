// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// clang-format on
#include <float.h>
#include <torch/extension.h>
#include <stdexcept>
#include "utils.h"
#include "xpu_types.hpp"
// #include "dtype_bfloat16.dp.hpp"
// #include "dtype_float16.dp.hpp"
#include "dtype_float32.dp.hpp"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

template <typename T>
struct Float_Trait {
  using Type = T;
};

template <>
struct Float_Trait<c10::Half> {
  using Type = uint16_t;
};

template <>
struct Float_Trait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

namespace vllm {

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(
    const Vec (&q)[N],
    const Vec (&k)[N],
    const sycl::nd_item<3>& item_ct1) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:0: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    qk += dpct::experimental::permute_sub_group_by_xor(
        0xffffffff, item_ct1.get_sub_group(), qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(
      const Vec (&q)[N],
      const Vec (&k)[N],
      const sycl::nd_item<3>& item_ct1) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k, item_ct1);
  }
};

template <
    typename scalar_t,
    typename scalar_sycl_t,
    int HEAD_SIZE,
    int BLOCK_SIZE>
struct paged_attention_xpu_v1_impl_ {
  static void call(
      scalar_t* __restrict__ out, // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                            // head_size/x, block_size, x]
      const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                            // head_size, block_size]
      const int num_kv_heads,
      const float scale,
      const int* __restrict__ block_tables, // [num_seqs,
                                            // max_num_blocks_per_seq]
      const int* __restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes, // [num_heads]
      const int q_stride,
      const int kv_block_stride,
      const int kv_head_stride,
      const int num_seqs,
      const int num_heads,
      const int num_blocks) {
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;
    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    sycl::queue& task_q = vllm::xpu::vllmGetQueue();
    sycl::buffer<scalar_sycl_t, 1> out_buf(
        (scalar_sycl_t*)out, num_seqs * num_heads * HEAD_SIZE);
    sycl::buffer<scalar_sycl_t, 1> q_buf(
        (scalar_sycl_t*)q, num_seqs * q_stride);
    sycl::buffer<int, 1> context_lens_buf(context_lens, num_seqs);
    sycl::buffer<int, 1> block_tables_buf(
        block_tables, num_seqs * max_num_blocks_per_seq);
    sycl::buffer<scalar_sycl_t, 1> k_cache_buf(
        (scalar_sycl_t*)k_cache, num_blocks * kv_block_stride);
    sycl::buffer<scalar_sycl_t, 1> v_cache_buf(
        (scalar_sycl_t*)v_cache, num_blocks * kv_block_stride);

    auto e0 = task_q.memset(
        out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));

    size_t logits_stride = num_heads * max_context_len_padded;
    size_t logits_bytes = num_seqs * logits_stride * sizeof(float);
    float* logits = (float*)sycl::aligned_alloc_device(
        64, logits_bytes, task_q.get_device(), task_q.get_context());
    sycl::event reset_logits = task_q.memset(logits, 0, logits_bytes);
    reset_logits.wait();
    auto e1 = task_q.submit([&](auto& h) {
      sycl::accessor q_acc(q_buf, h, sycl::read_only);
      sycl::accessor k_cache_acc(k_cache_buf, h, sycl::read_only);
      sycl::accessor context_lens_acc(context_lens_buf, h, sycl::read_only);
      sycl::accessor block_tables_acc(block_tables_buf, h, sycl::read_only);
      h.parallel_for(sycl::range(num_seqs, num_heads), [=](sycl::item<2> item) {
        size_t seq_idx = item[0];
        size_t head_idx = item[1];
        int context_len = context_lens_acc[seq_idx];
        const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
          const int32_t physical_block_idx =
              block_tables_acc[block_idx + max_num_blocks_per_seq * seq_idx];
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          size_t q_base_offset = seq_idx * q_stride + head_idx * HEAD_SIZE;
          size_t k_base_offset = physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride; // dim0,dim1
          float* __restrict__ head_block_logits = logits +
              seq_idx * logits_stride + head_idx * max_context_len_padded +
              block_idx * BLOCK_SIZE;
          for (int x_idx = 0; x_idx < HEAD_SIZE / x; ++x_idx) {
            for (int token_idx = 0; token_idx < BLOCK_SIZE; ++token_idx) {
              for (int i = 0; i < x; ++i) {
                head_block_logits
                    [token_idx] += (float)q_acc[i + x_idx * x + q_base_offset] *
                    (float)k_cache_acc[i + token_idx * x +
                                       BLOCK_SIZE * x_idx * x + k_base_offset] *
                    scale;
              }
            }
          }
        }
      });
    });
    e1.wait();

    auto e2 = task_q.submit([&](auto& h) {
      sycl::accessor context_lens_acc(context_lens_buf, h, sycl::read_only);
      h.parallel_for(sycl::range(num_seqs, num_heads), [=](sycl::item<2> item) {
        size_t seq_idx = item[0];
        size_t head_idx = item[1];
        int context_len = context_lens_acc[seq_idx];
        const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* head_logit_ptr = logits + seq_idx * logits_stride +
            head_idx * max_context_len_padded;
        float max_logit = head_logit_ptr[0];
        for (int i = 1; i < context_len; ++i) {
          max_logit =
              max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
        }
        float sum = 0.f;
        for (int i = 0; i < context_len; ++i) {
          float val = sycl::exp<float>(head_logit_ptr[i] - max_logit);
          head_logit_ptr[i] = val;
          sum += val;
        }
        const float inv_sum = 1.f / (sum + 1e-6f);
        for (int i = 0; i < context_len; ++i) {
          head_logit_ptr[i] *= inv_sum;
        }
        int remaining_seq_upper = block_num * BLOCK_SIZE;
        for (int i = context_len; i < remaining_seq_upper; ++i) {
          head_logit_ptr[i] = 0;
        }
      });
    });
    e2.wait();
    e0.wait();
    constexpr int head_partition_num = HEAD_SIZE / 16;
    auto e3 = task_q.submit([&](auto& h) {
      sycl::accessor output_acc(out_buf, h, sycl::read_write);
      sycl::accessor context_lens_acc(context_lens_buf, h, sycl::read_only);
      sycl::accessor v_cache_acc(v_cache_buf, h, sycl::read_only);
      sycl::accessor k_cache_acc(k_cache_buf, h, sycl::read_only);
      sycl::accessor block_tables_acc(block_tables_buf, h, sycl::read_only);

      h.parallel_for(
          sycl::range(num_seqs, num_heads, head_partition_num),
          [=](sycl::item<3> item) {
            size_t seq_idx = item[0];
            size_t head_idx = item[1];
            size_t head_part_idx = item[2];
            int context_len = context_lens_acc[seq_idx];
            const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int block_idx = 0; block_idx < block_num; ++block_idx) {
              const int32_t kv_head_idx = head_idx / num_queries_per_kv;
              const int32_t physical_block_idx = block_tables_acc
                  [block_idx + max_num_blocks_per_seq * seq_idx];
              const float* __restrict__ prob_vec_ptr = logits +
                  seq_idx * logits_stride + head_idx * max_context_len_padded +
                  block_idx * BLOCK_SIZE;
              size_t v_base_offset = physical_block_idx * kv_block_stride +
                  kv_head_idx * kv_head_stride +
                  BLOCK_SIZE * head_part_idx * 16;
              size_t out_base_offset = seq_idx * num_heads * HEAD_SIZE +
                  head_idx * HEAD_SIZE + head_part_idx * 16;
              for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                  output_acc[i + out_base_offset] +=
                      (scalar_sycl_t)(prob_vec_ptr[j] * (float)v_cache_acc[j + i * BLOCK_SIZE + v_base_offset]);
                }
              }
            }
          });
    });

    e3.wait();
    sycl::free(logits, task_q);
  };
};

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_xpu_v1_impl {
  static void call(
      scalar_t* __restrict__ out, // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
      const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                            // head_size/x, block_size, x]
      const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                            // head_size, block_size]
      const int num_kv_heads,
      const float scale,
      const int* __restrict__ block_tables, // [num_seqs,
                                            // max_num_blocks_per_seq]
      const int* __restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes, // [num_heads]
      const int q_stride,
      const int kv_block_stride,
      const int kv_head_stride,
      const int num_seqs,
      const int num_heads,
      const int num_blocks) {
    paged_attention_xpu_v1_impl_<scalar_t, scalar_t, HEAD_SIZE, BLOCK_SIZE>::
        call(
            out,
            q,
            k_cache,
            v_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            max_num_blocks_per_seq,
            alibi_slopes,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            num_seqs,
            num_heads,
            num_blocks);
  }
};

template <int NUM_WARPS>
inline float block_sum(
    float* red_smem,
    float sum,
    const sycl::nd_item<3>& item_ct1) {
  // Decompose the thread index into warp / lane.
  int warp = item_ct1.get_local_id(2) / WARP_SIZE;
  int lane = item_ct1.get_local_id(2) % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:2: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:42: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::experimental::permute_sub_group_by_xor(
        0xffffffff, item_ct1.get_sub_group(), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:3: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:43: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sum += dpct::experimental::permute_sub_group_by_xor(
        0xffffffff, item_ct1.get_sub_group(), sum, mask);
  }

  // Broadcast to other threads.
  /*
  DPCT1023:4: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group.
  */
  /*
  DPCT1096:44: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  return dpct::experimental::select_from_sub_group(
      0xffffffff, item_ct1.get_sub_group(), sum, 0);
}

template <
    typename scalar_t,
    typename Q_Vec_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int VEC_SIZE,
    int PARTITION_SIZE = 0> // Zero means no partitioning.
void paged_attention_kernel(
    float* __restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out, // [num_seqs, num_heads, max_num_partitions,
                                // head_size]
    const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, // [num_heads]
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    Q_Vec_t* q_vecs,
    float* red_smem) {
  const int seq_idx = item_ct1.get_group(1);
  const int partition_idx = item_ct1.get_group(0);
  const int max_num_partitions = item_ct1.get_group_range(0);
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  Q_Vec_t* q_vecs_ptr = (Q_Vec_t*)q_vecs;
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = item_ct1.get_local_id(2);
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = item_ct1.get_group(2);
  const int num_heads = item_ct1.get_group_range(2);
  const int num_queries_per_kv = num_heads / num_kv_heads;

  const int kv_head_idx = head_idx / num_queries_per_kv;
  ;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.

  // constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)),
  // 1);

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const Q_Vec_t* q_ptr =
      (Q_Vec_t*)(q + seq_idx * q_stride + head_idx * HEAD_SIZE);

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset * THREAD_GROUP_SIZE + i] =
        *reinterpret_cast<const Q_Vec_t*>(q_ptr + vec_idx * VEC_SIZE);
  }
  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); // TODO(naed90): possible speedup if this is replaced with
                      // a memory wall right before we use q_vecs

  // Memory planning.
  auto shared_mem = (char*)dpct_local;
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

      Q_Vec_t k_vecs[NUM_VECS_PER_THREAD];
      Q_Vec_t q_vecs_new[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache +
            physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;

        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const Q_Vec_t*>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        q_vecs_new[j] = *reinterpret_cast<const Q_Vec_t*>(
            &q_vecs[thread_group_offset * THREAD_GROUP_SIZE + j]);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Q_Vec_t q_vec_[NUM_VECS_PER_THREAD] = q_vecs + thread_group_offset *
      // THREAD_GROUP_SIZE;
      float qk = scale *
          Qk_dot<scalar_t, THREAD_GROUP_SIZE>::
              template dot<Q_Vec_t, NUM_VECS_PER_THREAD>(
                     q_vecs_new, k_vecs, item_ct1);
      // Add the ALiBi bias if slopes are given.
      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : sycl::fmax(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    /*
    DPCT1023:9: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:38: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(
        qk_max,
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, item_ct1.get_sub_group(), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    /*
    DPCT1023:10: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    /*
    DPCT1096:39: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    qk_max = sycl::fmax(
        qk_max,
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, item_ct1.get_sub_group(), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  /*
  DPCT1023:11: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group.
  */
  /*
  DPCT1096:40: The right-most dimension of the work-group used in the SYCL
  kernel that calls this function may be less than "32". The function
  "dpct::select_from_sub_group" may return an unexpected result on the CPU
  device. Modify the size of the work-group to ensure that the value of the
  right-most dimension is a multiple of "32".
  */
  qk_max = dpct::experimental::select_from_sub_group(
      0xffffffff, item_ct1.get_sub_group(), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = sycl::exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum, item_ct1);

  // Compute softmax.
  const float inv_sum = 1.f / (exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
        seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    vllm::from_float(
        logits_vec,
        *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += vllm::dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      /*
      DPCT1023:12: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor.
      */
      /*
      DPCT1096:41: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      acc += dpct::experimental::permute_sub_group_by_xor(
          0xffffffff, item_ct1.get_sub_group(), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  /*
  DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        vllm::from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <
    typename scalar_t,
    typename Q_Vec_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int VEC_SIZE>
void paged_attention_v1_kernel(
    scalar_t* __restrict__ out, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, // [num_heads]
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local,
    Q_Vec_t* q_vecs,
    float* red_smem) {
  paged_attention_kernel<
      scalar_t,
      Q_Vec_t,
      HEAD_SIZE,
      BLOCK_SIZE,
      NUM_THREADS,
      VEC_SIZE>(
      /* exp_sums */ nullptr,
      /* max_logits */ nullptr,
      out,
      q,
      k_cache,
      v_cache,
      num_kv_heads,
      scale,
      block_tables,
      context_lens,
      max_num_blocks_per_seq,
      alibi_slopes,
      q_stride,
      kv_block_stride,
      kv_head_stride,
      item_ct1,
      dpct_local,
      q_vecs,
      red_smem);
}

template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_xpu_v1_impl<c10::Half, HEAD_SIZE, BLOCK_SIZE> {
  static void call(
      c10::Half* __restrict__ out, // [num_seqs, num_heads, head_size]
      const c10::Half* __restrict__ q, // [num_seqs, num_heads, head_size]
      const c10::Half* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
      const c10::Half* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
      const int num_kv_heads,
      const float scale,
      const int* __restrict__ block_tables, // [num_seqs,
                                            // max_num_blocks_per_seq]
      const int* __restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes, // [num_heads]
      const int q_stride,
      const int kv_block_stride,
      const int kv_head_stride,
      const int num_seqs,
      const int num_heads,
      const int num_blocks) {
    paged_attention_xpu_v1_impl_<c10::Half, sycl::half, HEAD_SIZE, BLOCK_SIZE>::
        call(
            out,
            q,
            k_cache,
            v_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            max_num_blocks_per_seq,
            alibi_slopes,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            num_seqs,
            num_heads,
            num_blocks);
  }
};

template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_xpu_v1_impl<c10::BFloat16, HEAD_SIZE, BLOCK_SIZE> {
  static void call(
      c10::BFloat16* __restrict__ out, // [num_seqs, num_heads, head_size]
      const c10::BFloat16* __restrict__ q, // [num_seqs, num_heads, head_size]
      const c10::BFloat16* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                                 // head_size/x, block_size,
                                                 // x]
      const c10::BFloat16* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                                 // head_size, block_size]
      const int num_kv_heads,
      const float scale,
      const int* __restrict__ block_tables, // [num_seqs,
                                            // max_num_blocks_per_seq]
      const int* __restrict__ context_lens, // [num_seqs]
      const int max_num_blocks_per_seq,
      const float* __restrict__ alibi_slopes, // [num_heads]
      const int q_stride,
      const int kv_block_stride,
      const int kv_head_stride,
      const int num_seqs,
      const int num_heads,
      const int num_blocks) {
    paged_attention_xpu_v1_impl_<
        c10::BFloat16,
        sycl::ext::oneapi::bfloat16,
        HEAD_SIZE,
        BLOCK_SIZE>::
        call(
            out,
            q,
            k_cache,
            v_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            max_num_blocks_per_seq,
            alibi_slopes,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            num_seqs,
            num_heads,
            num_blocks);
  }
};

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)      \
  paged_attention_xpu_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call( \
      out_ptr,                                                 \
      query_ptr,                                               \
      key_cache_ptr,                                           \
      value_cache_ptr,                                         \
      num_kv_heads,                                            \
      scale,                                                   \
      block_tables_ptr,                                        \
      context_lens_ptr,                                        \
      max_num_blocks_per_seq,                                  \
      alibi_slopes_ptr,                                        \
      q_stride,                                                \
      kv_block_stride,                                         \
      kv_head_stride,                                          \
      num_seqs,                                                \
      num_heads,                                               \
      num_blocks);

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  queue.submit([&](sycl::handler& cgh) {                                    \
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                    \
        sycl::range<1>(shared_mem_size), cgh);                              \
    sycl::local_accessor<Q_vec, 1> q_vecs_acc_ct1(                          \
        sycl::range<1>(THREAD_GROUP_SIZE * num_vecs_per_thread), cgh);      \
    sycl::local_accessor<float, 1> red_smem_acc_ct1(                        \
        sycl::range<1>(2 * NUM_WARPS), cgh);                                \
                                                                            \
    auto out_ptr_ct0 = out_ptr;                                             \
    auto query_ptr_ct1 = query_ptr;                                         \
    auto key_cache_ptr_ct2 = key_cache_ptr;                                 \
    auto value_cache_ptr_ct3 = value_cache_ptr;                             \
    auto scale_ct5 = scale;                                                 \
    auto block_tables_ptr_ct6 = block_tables_ptr;                           \
    auto context_lens_ptr_ct7 = context_lens_ptr;                           \
    auto max_num_blocks_per_seq_ct8 = max_num_blocks_per_seq;               \
    auto alibi_slopes_ptr_ct9 = alibi_slopes_ptr;                           \
    auto q_stride_ct10 = q_stride;                                          \
    auto kv_block_stride_ct11 = kv_block_stride;                            \
    auto kv_head_stride_ct12 = kv_head_stride;                              \
                                                                            \
    cgh.parallel_for(                                                       \
        sycl::nd_range<3>(grid * block, block),                             \
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] { \
          paged_attention_v1_kernel<                                        \
              T,                                                            \
              Q_vec,                                                        \
              HEAD_SIZE,                                                    \
              BLOCK_SIZE,                                                   \
              NUM_THREADS,                                                  \
              VEC_SIZE>(                                                    \
              out_ptr_ct0,                                                  \
              query_ptr_ct1,                                                \
              key_cache_ptr_ct2,                                            \
              value_cache_ptr_ct3,                                          \
              num_kv_heads,                                                 \
              scale_ct5,                                                    \
              block_tables_ptr_ct6,                                         \
              context_lens_ptr_ct7,                                         \
              max_num_blocks_per_seq_ct8,                                   \
              alibi_slopes_ptr_ct9,                                         \
              q_stride_ct10,                                                \
              kv_block_stride_ct11,                                         \
              kv_head_stride_ct12,                                          \
              item_ct1,                                                     \
              dpct_local_acc_ct1.get_pointer(),                             \
              q_vecs_acc_ct1.get_pointer(),                                 \
              red_smem_acc_ct1.get_pointer());                              \
        });                                                                 \
  });

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128>
void paged_attention_xpu_v1_impl_launcher(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using FloatType = typename Float_Trait<T>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;

  int num_vecs_per_thread = 1; // FIXME
  assert(head_size % THREAD_GROUP_SIZE == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes
      ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
      : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  /*
  DPCT1083:22: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);
  std::cout
  << "NUM_THREADS is " << NUM_THREADS
  << " num_seqs is " << num_seqs
  << " num_heads is " << num_heads
  << " BLOCK_SIZE is " << BLOCK_SIZE
  << " THREAD_GROUP_SIZE is " << THREAD_GROUP_SIZE
  << " VEC_SIZE is " << VEC_SIZE
  << " max_context_len is " << max_context_len
  << " padded_max_context_len is " << padded_max_context_len
  << " WARP_SIZE is " << WARP_SIZE
  << " NUM_WARPS is " << NUM_WARPS
  << " shared_mem_size is " << shared_mem_size << std::endl;

  sycl::range<3> grid(1, num_seqs, num_heads);
  sycl::range<3> block(1, 1, NUM_THREADS);
  sycl::queue& queue = vllm::xpu::vllmGetQueue();

  // using K_vec = sycl::float2;
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
  queue.wait();
}

// template <typename T, int BLOCK_SIZE>
// void paged_attention_xpu_v1_impl_launcher(
//     torch::Tensor& out,
//     torch::Tensor& query,
//     torch::Tensor& key_cache,
//     torch::Tensor& value_cache,
//     int num_kv_heads,
//     float scale,
//     torch::Tensor& block_tables,
//     torch::Tensor& context_lens,
//     int max_context_len,
//     const c10::optional<torch::Tensor>& alibi_slopes) {
//   int num_seqs = query.size(0);
//   int num_heads = query.size(1);
//   int head_size = query.size(2);
//   int max_num_blocks_per_seq = block_tables.size(1);
//   int q_stride = query.stride(0);
//   int kv_block_stride = key_cache.stride(0);
//   int kv_head_stride = key_cache.stride(1);
//   int num_blocks = key_cache.size(0);

//   // NOTE: alibi_slopes is optional.
//   const float* alibi_slopes_ptr = alibi_slopes
//       ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
//       : nullptr;

//   T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
//   T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
//   T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
//   T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
//   int* block_tables_ptr = block_tables.data_ptr<int>();
//   int* context_lens_ptr = context_lens.data_ptr<int>();

//   switch (head_size) {
//     case 64:
//       LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
//       break;
//     case 80:
//       LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
//       break;
//     case 96:
//       LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
//       break;
//     case 112:
//       LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
//       break;
//     case 128:
//       LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
//       break;
//     case 256:
//       LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
//       break;
//     default:
//       TORCH_CHECK(false, "Unsupported head size: ", head_size);
//       break;
//   }
// }

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                  \
  vllm::paged_attention_xpu_v1_impl_launcher<T, BLOCK_SIZE>( \
      out,                                                   \
      query,                                                 \
      key_cache,                                             \
      value_cache,                                           \
      num_kv_heads,                                          \
      scale,                                                 \
      block_tables,                                          \
      context_lens,                                          \
      max_context_len,                                       \
      alibi_slopes);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                        \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_KERNEL_LAUNCHER(T, 16);                                \
      break;                                                      \
    case 32:                                                      \
      CALL_KERNEL_LAUNCHER(T, 32);                                \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

} // namespace vllm

void paged_attention_v1_xpu(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "paged_attention_xpu_v1_impl", [&] {
        CALL_KERNEL_LAUNCHER_BLOCK_SIZE(scalar_t);
      });
}

void paged_attention_v2_xpu(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes) {
  TORCH_CHECK(false, "paged_attention_v2 is unsupported on CPU.")
}