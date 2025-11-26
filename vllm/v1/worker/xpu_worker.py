# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()
        # this API is not ready in torch 2.9, but ready in torch 2.10
        # see https://github.com/pytorch/pytorch/pull/156812
        torch.cuda.mem_get_info = torch.xpu.mem_get_info
        torch.cuda.stream = torch.xpu.stream

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        self.profiler: Any | None = None
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    # we provide this function due to `torch.xpu.mem_get_info()` doesn't
    # return correct free_gpu_memory on intel client GPU. We need to
    # calculate/estiamte it.
    def xpu_get_mem_info(self):
        if current_platform.is_data_center_gpu():
            return torch.xpu.mem_get_info()
        else:
            _, total_gpu_memory = torch.xpu.mem_get_info()
            # FIXME: memory_allocated() doesn't count non-torch allocations,
            # and we don't have any API to get it. so we mark it as 128MB.
            used_memory = torch.xpu.memory_allocated()
            non_torch_allocations = 128 * 1024 * 1024
            free_gpu_memory = total_gpu_memory - (used_memory + non_torch_allocations)
            return free_gpu_memory, total_gpu_memory

    # def init_device(self):
    #     device = self.device_config.device
    #     if (
    #         isinstance(device, torch.device)
    #         and device.type == "xpu"
    #         and current_platform.is_xpu()
    #     ):
    #         self.device = torch.device(f"xpu:{self.local_rank}")
    #         current_platform.set_device(self.device)
    #         current_platform.check_if_supports_dtype(self.model_config.dtype)
    #         torch.xpu.empty_cache()
    #         self.init_gpu_memory = torch.xpu.get_device_properties(
    #             self.local_rank
    #         ).total_memory
    #     else:
    #         raise RuntimeError(f"Not support device type")

    #     ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
    #     ENV_LOCAL_WORLD_SIZE = os.getenv(
    #         "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
    #     )
    #     os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
    #     os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
    #     os.environ["LOCAL_RANK"] = str(self.local_rank)

    #     init_worker_distributed_environment(
    #         self.vllm_config,
    #         self.rank,
    #         self.distributed_init_method,
    #         self.local_rank,
    #         current_platform.dist_backend,
    #     )

    #     # Set random seed.
    #     set_random_seed(self.model_config.seed)
    #     #       take current memory snapshot
    #     self.init_snapshot = MemorySnapshot()
    #     self.requested_memory = (
    #         self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
    #     )
    #     if self.init_snapshot.free_memory < self.requested_memory:
    #         GiB = lambda b: round(b / GiB_bytes, 2)
    #         raise ValueError(
    #             f"Free memory on device "
    #             f"({GiB(self.init_snapshot.free_memory)}/"
    #             f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
    #             f"is less than desired GPU memory utilization "
    #             f"({self.cache_config.gpu_memory_utilization}, "
    #             f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
    #             f"utilization or reduce GPU memory used by other processes."
    #         )
    #     # Construct the model runner
    #     self.model_runner = XPUModelRunner(  # type: ignore
    #         self.vllm_config, self.device
    #     )
