import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        # 最大并发请求数
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        # 记录"逻辑上的第 N 个 Token,存在物理上的哪个 Block 里".
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # 暂存每个请求实际的文本 Token ID.
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
