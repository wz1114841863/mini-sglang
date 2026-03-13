# 延迟评估类型注解, 以避免循环导入问题
from __future__ import annotations

# 自动生成构造函数, 并且使实例不可变(frozen=True)
from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # 允许的最大扩展长度, 包括输入和输出的总长度
    max_extend_tokens: int = 8192
    # 缓存机制: "radix"表示使用基数树进行缓存, "lru"表示使用最近最少使用算法进行缓存
    cache_type: str = "radix"
    # 是否启用离线模式, 离线模式下调度器不会与后端通信, 适用于测试和调试
    offline_mode: bool = False

    # networking config
    # field(default_factory=...) 确保每次创建一个新的实例时都会调用_get_pid_suffix函数,
    # 生成一个唯一的后缀, 避免不同实例之间的IPC地址冲突
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """后端引擎的通信地址"""
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """用于将 Token 转换回文字的解码器地址"""
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """: 调度器向其他组件广播消息的地址"""
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
