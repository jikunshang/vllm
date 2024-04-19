#include "cpu/cpu_types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace {
enum class RankStat : char { READY = 0, EXECUTE, DONE };

struct SHMContext {
  volatile RankStat rank_stats[64];
  int group_size;
  size_t rank_buffer_size;
  char _padding[48];

  SHMContext(const int group_size, const size_t rank_buffer_size)
      : group_size(group_size), rank_buffer_size(rank_buffer_size) {
    static_assert(sizeof(SHMContext) % 64 == 0);
    TORCH_CHECK(group_size <= 64);
    TORCH_CHECK((size_t)this % 64 == 0);
    for (int i = 0; i < 64; ++i) {
      rank_stats[i] = RankStat::DONE;
    }
  }

  template <typename T> T *rank_ptr(int rank) {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(this + 1) +
                                 rank_buffer_size * rank);
  }

  RankStat get_rank_stat(int rank) const { return rank_stats[rank]; }

  void set_rank_stat(int rank, RankStat stat) { rank_stats[rank] = stat; }

  bool is_all_done() const {
    for (int i = 0; i < group_size; ++i) {
      if (get_rank_stat(i) != RankStat::DONE) {
        return false;
      }
    }
    return true;
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "SHMContext: "
       << "\ngroup_size: " << group_size
       << "\nrank_buffer_size: " << rank_buffer_size << "\nrank_stat: [";
    for (int i = 0; i < group_size; ++i) {
      switch (rank_stats[i]) {
      case RankStat::READY:
        ss << "READY, ";
        break;
      case RankStat::EXECUTE:
        ss << "EXECUTE, ";
        break;
      case RankStat::DONE:
        ss << "DONE, ";
        break;
      default:
        TORCH_CHECK(false, "Invalid RankStat type.");
      }
    }
    ss << "]";

    return ss.str();
  }
};

namespace shm_cc_ops {
void parallel_memcpy(void *dst, void *src, size_t len) {
  int thread_num = omp_get_max_threads();
  size_t partition_len = (len + thread_num - 1) / thread_num;
  size_t tail_partition_len = len - partition_len * (thread_num - 1);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    std::memcpy((char *)dst + i * partition_len,
                (char *)src + i * partition_len,
                i == thread_num - 1 ? tail_partition_len : partition_len);
  }
}

void gather(SHMContext *ctx, int rank, void *data, size_t len) {
  TORCH_CHECK(len <= ctx->rank_buffer_size);
  if (rank == 0) {
    for (int i = 0; i < ctx->group_size; ++i) {
      ctx->set_rank_stat(i, RankStat::READY);
    }
  } else {
    while (ctx->get_rank_stat(rank) != RankStat::READY)
      _mm_pause();
  }
  ctx->set_rank_stat(rank, RankStat::EXECUTE);
  parallel_memcpy(ctx->rank_ptr<void>(rank), data, len);
  ctx->set_rank_stat(rank, RankStat::DONE);

  if (rank == 0) {
    while (!ctx->is_all_done())
      _mm_pause();
  }
}

void broadcast(SHMContext *ctx, int rank, void *data, size_t len) {
  if (rank == 0) {
    for (int i = 0; i < ctx->group_size; ++i) {
      ctx->set_rank_stat(i, RankStat::READY);
    }
  } else {
    while (ctx->get_rank_stat(rank) != RankStat::READY)
      _mm_pause();
  }
  ctx->set_rank_stat(rank, RankStat::EXECUTE);
  parallel_memcpy(data, ctx->rank_ptr<void>(0), len);
  ctx->set_rank_stat(rank, RankStat::DONE);
  if (rank == 0) {
    while (!ctx->is_all_done())
      _mm_pause();
  }
}

}; // namespace shm_cc_ops

class SHMManager {
public:
  explicit SHMManager(const std::string &ip_port, const int group_size,
                      const int rank, const size_t rank_buffer_size)
      : _rank(rank), _shm_name(""), _shared_mem_ptr(nullptr),
        _shm_ctx(nullptr) {
    _shm_name = get_shm_name(ip_port);
    _shared_mem_ptr = init_shm(group_size, rank_buffer_size);

    if (rank == 0) {
      _shm_ctx = new (_shared_mem_ptr)
          SHMContext(group_size, round_size(rank_buffer_size));
    } else {
      _shm_ctx = reinterpret_cast<SHMContext *>(_shared_mem_ptr);
    }
  }

  ~SHMManager() { destroy_shm(); }

  SHMContext *get_shm_ctx() const {
    return reinterpret_cast<SHMContext *>(_shared_mem_ptr);
  }

  static std::string get_shm_name(const std::string &ip_port) {
    return "/vllm_" + ip_port;
  }

private:
  static size_t round_size(const size_t size) {
    return ((size + 63) >> 6) << 6;
  }

  void *init_shm(const int group_size, const size_t rank_buffer_size) {
    const std::string &shm_name = _shm_name;
    const int local_rank = _rank;
    const size_t rounded_rank_buffer_size = round_size(rank_buffer_size);
    const size_t shm_size =
        sizeof(SHMContext) + group_size * rounded_rank_buffer_size;

    int fd = -1;
    if (local_rank == 0) {
      fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                    S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "create shm in SHMManager failed. errno: " +
                               std::to_string(errno));

      if (ftruncate(fd, shm_size) == -1)
        TORCH_CHECK(false, "ftruncate in SHMManager failed. errno: " +
                               std::to_string(errno));
    } else {
      fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "open shm in SHMManager failed. errno: " +
                               std::to_string(errno));
    }

    void *shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);

    if (shm_ptr == MAP_FAILED) {
      TORCH_CHECK(false,
                  "mmap in SHMManager failed. errno: " + std::to_string(errno));
    }

    TORCH_CHECK((size_t)shm_ptr % 64 == 0)

    return shm_ptr;
  }

  void destroy_shm() {
    if (!_shm_name.empty() && _shared_mem_ptr != nullptr) {
      shm_unlink(_shm_name.c_str());
    }
  }

  int _rank;
  std::string _shm_name;
  void *_shared_mem_ptr;
  SHMContext *_shm_ctx;
};

static std::unique_ptr<SHMManager> shm_manager_singleton = nullptr;

template <typename scalar_t>
void shm_allreduce_sum(SHMContext *ctx, const int rank, scalar_t *data,
                       size_t elem_num) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  TORCH_CHECK(elem_num % VEC_ELEM_NUM == 0);

  const size_t bytes = elem_num * sizeof(scalar_t);
  TORCH_CHECK(bytes <= ctx->rank_buffer_size);

  shm_cc_ops::gather(ctx, rank, data, bytes);

  if (rank == 0) {
    const int thread_num = omp_get_max_threads();
    const int vec_num = elem_num / VEC_ELEM_NUM;
    const int partition_vec_num = (vec_num + thread_num - 1) / thread_num;
    const int tail_partition_vec_num =
        vec_num - partition_vec_num * (thread_num - 1);
    for (int i = 1; i < ctx->group_size; ++i) {
#pragma omp parallel for schedule(static, 1)
      for (int j = 0; j < thread_num; ++j) {
        const int curr_thread_vec_num =
            (j == thread_num - 1) ? tail_partition_vec_num : partition_vec_num;
        scalar_t *rank_0_ptr =
            ctx->rank_ptr<scalar_t>(0) + j * partition_vec_num * VEC_ELEM_NUM;
        scalar_t *rank_i_ptr =
            ctx->rank_ptr<scalar_t>(i) + j * partition_vec_num * VEC_ELEM_NUM;
        for (int k = 0; k < curr_thread_vec_num; ++k) {
          scalar_vec_t rank_0_data(rank_0_ptr + k * VEC_ELEM_NUM);
          scalar_vec_t rank_i_data(rank_i_ptr + k * VEC_ELEM_NUM);
          vec_op::FP32Vec8 rank_0_data_fp32(rank_0_data);
          vec_op::FP32Vec8 rank_i_data_fp32(rank_i_data);
          scalar_vec_t result(rank_0_data_fp32 + rank_i_data_fp32);
          result.save(rank_0_ptr + k * VEC_ELEM_NUM);
        }
      }
    }
  }

  shm_cc_ops::broadcast(ctx, rank, data, bytes);
}

} // namespace

void shm_allreduce(torch::Tensor &data, int rank) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_allreduce_sum", [&] {
    CPU_KERNEL_GUARD_IN(shm_allreduce_sum)
    shm_allreduce_sum(shm_manager_singleton->get_shm_ctx(), rank,
                      data.data_ptr<scalar_t>(), data.numel());
    CPU_KERNEL_GUARD_OUT(shm_allreduce_sum)
  });
}

std::string init_shm_manager(const std::string &ip_port, const int group_size,
                             const int rank, const size_t rank_buffer_size) {
  if (shm_manager_singleton == nullptr) {
    shm_manager_singleton = std::make_unique<SHMManager>(
        ip_port, group_size, rank, rank_buffer_size);
    return shm_manager_singleton->get_shm_ctx()->to_string();
  } else {
    TORCH_CHECK(
        false,
        "Duplicate initialization of shm_manager_singleton is not allowed.")
  }
}