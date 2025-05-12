# SPDX-License-Identifier: Apache-2.0
"""
This file contains a new class `MooncakeStore` that allows developers to
think of KV cache transfer operations as putting new KV cache entries
into a remote KVStore-based lookup buffer and getting existing KV caches
from this remote lookup buffer.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        rank_id = torch.distributed.get_rank()
        rank_id = rank_id % 8
        # please check mlx interface name on your node!! 
        if rank_id == 0:
            device = "mlx5_0"
        else:
            device = "mlx5_" + str(rank_id+2)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 53687091200),
            local_buffer_size=config.get("local_buffer_size", 10737418240),
            protocol=config.get("protocol", "tcp"),
            device_name=device,
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore(KVLookupBufferBase):

    def __init__(
        self,
        config: VllmConfig,
    ):

        try:
            from mooncake import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        # This interface has not been implemented since it is incompatible with
        # the future layer-by-layer communication implementation of
        # MooncakeStoreConnector. MooncakeStoreConnector will send the key
        # cache, value cache, and hidden states separately and asynchronously
        # and use a message queue to notify their transfer states for the
        # decode instances and the proxy.
        raise NotImplementedError

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        # This interface has not been implemented since it is incompatible with
        # the future layer-by-layer communication implementation of
        # MooncakeStoreConnector. MooncakeStoreConnector will send the key
        # cache, value cache, and hidden states separately and asynchronously
        # and use a message queue to notify their transfer states for the
        # decode instances and the proxy.
        raise NotImplementedError

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def put(
        self,
        key: str,
        value: Optional[torch.Tensor],
    ) -> None:
        # A message queue needs to be introduced before making it asynchronous.
        if value is not None:
            self._put_impl(key, value)

    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        # A message queue needs to be introduced before making it asynchronous.
        value = self._get_impl(key)
        return value

    def _put_impl(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        """Put KVCache to Mooncake Store"""
        device_id = value.device.index if value.device.type == 'hpu' else -1
        logger.debug(f"putting, device id: {device_id}")
        device_tensor = torch.tensor(device_id, dtype=torch.int32, device="cpu")
        value = value.cpu()
        value_bytes = safetensors_save({
            "tensor": value,
            "device_id": device_tensor
        })
        try:
            self.store.put(key, value_bytes)
        except TypeError as err:
            logger.error("Failed to put value into Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_impl(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err

        if data:
            loaded_tensors = safetensors_load(data)
            tensor = loaded_tensors["tensor"]
            device_id_tensor = loaded_tensors["device_id"]
            device_id = int(device_id_tensor.item())
            device = torch.device(
                'hpu', device_id) if device_id >= 0 else torch.device('cpu')
            return tensor.to(device)

        return None

    def put_unsafe(
        self,
        key: str,
        value: Optional[torch.Tensor],
    ) -> None:
        """Put KVCache to Mooncake Store"""
        value = value.cpu()
        start_serde = time.time()
        data_ptr = value.data_ptr()
        element_size = value.element_size()
        numel = value.numel()
        total_size = element_size * numel
        end_serde = time.time()
        try:
            self.store.put_unsafe(key, data_ptr, total_size)
        except TypeError as err:
            logger.error("Failed to put value into Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err
        end_put = time.time()
        logger.info(f"contiguous time: {end_serde - start_serde}, put time: {end_put - end_serde}")


    def get_unsafe(self, key: str, shape, dtype) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store without type checking"""
        start_get = time.time()
        data = self.store.get(key)
        end_get = time.time()
        if data:
            tensor = torch.frombuffer(data, dtype=dtype)
            tensor = tensor.reshape(shape)
            end_from_buffer = time.time()
            logger.info(f"from buffer time: {end_from_buffer - end_get}, get time: {end_get - start_get}")
            return tensor
        return None