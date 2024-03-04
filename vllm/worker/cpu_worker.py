"""A CPU worker class."""
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker
from vllm.logger import init_logger

logger = init_logger(__name__) 

class Worker(Worker):
    """A worker class that executes (a partition of) the model on a CPU socket.

    Each worker is associated with a single CPU socket. The worker is responsible for
    maintaining the KV cache and executing the model on the CPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        assert device_config.device_type == "cpu"
        model_config = Worker._verify_and_get_model_config(model_config)
        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
        ) 

    def init_model(self, cupy_port: Optional[int] = None) -> None:
        self.rank = 0
        self.device = torch.device("cpu")
        self.init_gpu_memory = 0

        # Initialize the distributed environment.
        Worker.init_distributed_environment(
            self.parallel_config,
            self.rank,
            self.distributed_init_method,
        )
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # For CPU device, the block number will be calculated based on the swap_space.
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)
        num_gpu_blocks = 0
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # To re-use the cache management procedure, use cpu cache as 'gpu cache'.
        return num_cpu_blocks, num_gpu_blocks


    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config,
                                        self.device_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)


    def cache_copy(
        self,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)


    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            assert len(blocks_to_swap_in) == 0 
            assert len(blocks_to_swap_out) == 0
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_copy(blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        return output

    @staticmethod
    def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
        if (config.dtype == torch.float16):
            logger.warning(f"float16 is not supported not CPU, casting to bfloat16.")
            config.dtype = torch.bfloat16
        if (config.enforce_eager == False):
            logger.warning(f"CUDA graph is not supported on CPU, fallback to the eager mode.")
            config.enforce_eager = True
        return config

    @staticmethod
    def init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        """Initialize the distributed environment."""
        def all_reduce_warmup():
            torch.distributed.all_reduce(torch.zeros(1).cpu())

        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {parallel_config.world_size}).")
        elif not distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            backend = "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                world_size=parallel_config.world_size,
                rank=rank,
                init_method=distributed_init_method,
            )

        # A small all_reduce for warmup.
        all_reduce_warmup()

        ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                        parallel_config.pipeline_parallel_size)
