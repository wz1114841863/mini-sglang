from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    # 模型路径或 HuggingFace ID
    model_path: str
    # 张量并行信息
    tp_info: DistributedInfo
    # 数据类型
    dtype: torch.dtype
    # 最大运行请求数
    max_running_req: int = 256
    # 注意力后端和 MoE 后端,"auto" 表示自动选择
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # CUDA 图相关配置
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    # KV 缓存管理相关配置
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    # 是否使用 dummy 权重(全零权重),主要用于测试和调试,能显著加快模型加载速度
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
