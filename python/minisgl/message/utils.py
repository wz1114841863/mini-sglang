from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """递归地序列化一个对象, 将其转换成可以通过JSON序列化的类型."""
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    """负责将非基础对象转化为字典."""
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """递归地反序列化一个对象, 将其从JSON序列化的类型转换回原始对象."""
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """负责将字典转换回原始对象."""
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)


if __name__ == "__main__":
    # 1. 定义一个简单的自定义类用于测试
    class UserConfig:
        def __init__(self, name: str, scores: torch.Tensor, metadata: Dict):
            self.name = name
            self.scores = scores
            self.metadata = metadata

        def __repr__(self):
            return f"UserConfig(name={self.name}, scores={self.scores}, metadata={self.metadata})"

    # 2. 准备类映射表 (反序列化时必须)
    # 键是类名字符串,值是类本身
    class_map = {"UserConfig": UserConfig}

    # 3. 创建一个复杂的嵌套对象
    original_obj = UserConfig(
        name="Test",
        scores=torch.tensor([1.5, 2.0, 3.5], dtype=torch.float32),
        metadata={"id": 101, "tags": ["AI", "Serialization"], "version": 1.0},
    )

    print("--- 原始对象 ---")
    print(original_obj)

    # 4. 执行序列化
    serialized_data = serialize_type(original_obj)
    print("\n--- 序列化后的字典 (可以转为 JSON) ---")
    import pprint

    pprint.pprint(serialized_data)

    # 5. 执行反序列化
    # 模拟从字典恢复对象
    recovered_obj = deserialize_type(class_map, serialized_data)

    print("\n--- 反序列化后的对象 ---")
    print(recovered_obj)

    # 6. 验证正确性
    assert recovered_obj.name == original_obj.name
    assert torch.equal(recovered_obj.scores, original_obj.scores)
    assert recovered_obj.metadata == original_obj.metadata
    print("\n✅ 测试通过:序列化与反序列化数据一致!")
