from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """维护当前正在并行生成的请求集合, 计算它们还需要消耗多少显存资源."""

    page_size: int
    # 存放着所有已经完成了 Prefill阶段, 但还没有完成 Decode 阶段的请求
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """将新的请求加入到正在运行的请求集合中, 但只保留那些可以进入 Decode阶段的请求."""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """将一个请求从正在运行的请求集合中移除, 这通常发生在一个请求完成了 Decode阶段之后."""
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """根据请求的唯一标识符 (uid) 从正在运行的请求集合中移除一个请求, 这通常发生在一个请求被用户取消或者发生错误时."""
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """资源估算.系统需要知道这批正在生成的请求,未来最高可能还会消耗多少显存."""
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """直接把当前 running_reqs 里所有的请求打包成一个大 Batch"""
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
