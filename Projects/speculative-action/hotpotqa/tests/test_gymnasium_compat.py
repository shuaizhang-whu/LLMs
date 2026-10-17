"""Gymnasium 1.x wrapper 属性访问的兼容性回归测试。"""

import unittest

import gymnasium as gym

from src.wrappers import HotPotQAWrapper
from src.wrappers import HistoryWrapper, LoggingWrapper


class DummyWikiEnv(gym.Env):
    """只提供 HotPotQAWrapper._get_info 所需状态的最小环境。"""

    def __init__(self):
        self.steps = 2
        self.answer = None

    def reset(self, seed=None, return_info=False, options=None):
        return "reset"

    def step(self, action, step_type="wiki"):
        return "observation", 0, False, {"steps": self.steps, "answer": self.answer}


class GymnasiumCompatibilityTest(unittest.TestCase):
    def test_hotpot_wrapper_reads_inner_state_via_public_api(self):
        wrapper = HotPotQAWrapper(DummyWikiEnv(), split="dev")
        wrapper.data_idx = 0

        info = wrapper._get_info()

        self.assertEqual(info["steps"], 2)
        self.assertIsNone(info["answer"])

    def test_history_wrapper_preserves_requested_dataset_index(self):
        wrapped = HotPotQAWrapper(DummyWikiEnv(), split="dev")
        wrapped = LoggingWrapper(wrapped, file_id="test")
        wrapped = HistoryWrapper(wrapped, obs_format="history")

        observation = wrapped.reset(seed=248, idx=7107)

        self.assertEqual(wrapped.get_wrapper_attr("data_idx"), 7107)
        self.assertEqual(
            observation,
            f"Question: {wrapped.get_wrapper_attr('data')[7107][0]}",
        )


if __name__ == "__main__":
    unittest.main()
