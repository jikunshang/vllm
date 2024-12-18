import time
import torch

from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import xpu_graph_capture
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.v1.attention.backends.ipex_attn import IPEXAttentionBackend
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        input_registry: InputRegistry = INPUT_REGISTRY,
    ):
        super().__init__(vllm_config, device, input_registry)
        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # self._dummy_run(self.model, self.max_num_tokens)
        torch.xpu.synchronize()

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = IPEXAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))
    def capture_model(self):
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping XPU graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return
        start_time = time.perf_counter()
        start_used_memory = torch.xpu.memory_allocated()
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with xpu_graph_capture():
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(self.model, num_tokens, self.kv_caches)
                self._dummy_run(self.model, num_tokens, self.kv_caches)
        end_time = time.perf_counter()
        end_used_memory = torch.xpu.memory_allocated()
        elapsed_time = end_time - start_time
        cuda_graph_size = end_used_memory - start_used_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
