"""Use the real local Qwen3.5 tokenizer; never send a model request."""
import copy
import unittest
from unittest.mock import patch

from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer

from scripts.run_qwen35_expanded import WARMUP_USER_TEXT, warmup_prompt_ids
from spork_core.interactive_successor import SuccessorInteractiveAgent
from tau2.registry import registry


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/data/zhangshuai/LLMs/Models/Qwen3.5-9B", local_files_only=True)

    def test_actual_template_warmup_for_both_domains_and_all_arms(self):
        for domain in ("airline", "retail"):
            env = registry.get_env_constructor(domain)(solo_mode=False)
            seen = []
            for mode in ("spork", "raw", "raw_no_probe", "raw_one_probe"):
                with self.subTest(domain=domain, mode=mode):
                    agent = SuccessorInteractiveAgent(env, None, self.tokenizer, mode)
                    state = agent.get_init_state()
                    original = copy.deepcopy(state)
                    # Reproduce the real failure, rather than testing a mock template.
                    with self.assertRaisesRegex(TemplateError, "No user query"):
                        agent.adapter.render_main_prompt(state, enable_thinking=True)
                    with patch.object(agent, "get_init_state", return_value=state):
                        ids = warmup_prompt_ids(agent)
                    self.assertEqual(state, original)
                    self.assertTrue(ids)
                    self.assertTrue(all(type(t) is int for t in ids))
                    self.assertLess(len(ids) + 1, 40960)
                    seen.append(ids)
            self.assertTrue(all(ids == seen[0] for ids in seen))

    def test_warmup_is_separate_from_the_first_real_user_message(self):
        env = registry.get_env_constructor("airline")(solo_mode=False)
        agent = SuccessorInteractiveAgent(env, None, self.tokenizer, "spork")
        state = agent.get_init_state()
        with patch.object(agent, "get_init_state", return_value=state):
            warm_ids = warmup_prompt_ids(agent)
        state.append({"role": "user", "content": "Please check my reservation."})
        prompt = agent.adapter.render_main_prompt(state, enable_thinking=True)
        self.assertNotIn(WARMUP_USER_TEXT, prompt)
        self.assertIn("Please check my reservation.", prompt)
        self.assertNotEqual(self.tokenizer.encode(prompt, add_special_tokens=False), warm_ids)


if __name__ == "__main__":
    unittest.main()
