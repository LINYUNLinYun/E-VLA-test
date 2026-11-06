from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


class TestPolicy:
    """
    最简单的 LIBERO 测试 policy。

    LIBERO 默认使用 7 维 action:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]

    在默认 OSC_POSE controller 下，位置 action 的输入范围是 [-1, 1]，
    其中 1.0 对应最大约 +0.05 m 的位置增量。

    因此这里连续输出 10 次:
        [1, 0, 0, 0, 0, 0, 0]

    即要求末端执行器每次沿 +x 方向移动约 5 cm。
    """

    def __init__(self, num_steps: int = 10, gripper_action: float = 0.0) -> None:
        if gripper_action not in (-1.0, 0.0, 1.0):
            raise ValueError("gripper_action 必须是 -1、0 或 1")
        self.num_steps = num_steps
        self.gripper_action = gripper_action
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def get_action(self, obs: Optional[Any] = None) -> FloatArray:
        """
        Args:
            obs:
                LIBERO 环境返回的 observation。
                这个测试 policy 不使用 observation。

        Returns:
            action: np.ndarray, shape=(7,)
        """
        if self.step_count < self.num_steps:
            action = np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_action],
                dtype=np.float32,
            )
        else:
            action = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_action],
                dtype=np.float32,
            )

        self.step_count += 1
        return action


if __name__ == "__main__":
    policy = TestPolicy(num_steps=10)

    for i in range(12):
        action = policy.get_action()
        print(f"step {i:02d}: {action}")
