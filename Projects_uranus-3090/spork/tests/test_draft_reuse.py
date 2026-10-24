import unittest
from types import SimpleNamespace

import numpy as np

from spork_core.draft_reuse import DraftBank, ReuseBridge
from scripts.replay_draft_reuse import simulate


class DraftReuseTests(unittest.TestCase):
    def test_resumes_after_wrong_argument_only_when_visible_suffix_matches(self):
        draft = [1, 2, 3, 4, 5, 60, 7, 8, 9, 10, 11, 12, 13]
        body = [1, 2, 3, 4, 5, 99, 7, 8, 9, 10, 11]
        for mode in ("single", "rolling", "bank_prefix", "bank_suffix"):
            bank = DraftBank(mode, min_suffix=5)
            bank.add(draft)
            proposal = bank.propose(body, 20)
            self.assertEqual(proposal.tokens, (12, 13) if mode == "bank_suffix" else ())

    def test_retention_is_distinct_from_repeated_latest_draft(self):
        for mode in ("rolling", "bank_prefix"):
            bank = DraftBank(mode)
            bank.add([1, 2, 3, 4])
            bank.add([9, 8, 7])
            self.assertEqual(bank.propose([1, 2], 1).tokens, (3,) if mode == "bank_prefix" else ())

    def test_single_does_not_refire_until_new_registration(self):
        bank = DraftBank("single")
        bank.add([1, 2, 3, 4])
        self.assertEqual(bank.propose([], 2).tokens, (1, 2))
        self.assertFalse(bank.propose([1, 2], 2).tokens)
        bank.add([1, 2, 3, 4])
        self.assertEqual(bank.propose([1, 2], 2).tokens, (3, 4))

    def test_no_proposal_after_tool_end_or_without_budget(self):
        bank = DraftBank()
        bank.add([1, 2, 3, 4, 5, 6])
        self.assertFalse(bank.propose([], 0).tokens)
        self.assertFalse(bank.propose([151658, 1, 2, 3, 4, 5], 20).tokens)

    def test_worker_isolation_accounting_and_cleanup(self):
        rid, other = "cmpl-a-0-uuid", "cmpl-b-0-uuid"
        def propose(num_speculative_tokens, sampled_token_ids, *args, **kwargs):
            return [[88], [89]]
        runner = SimpleNamespace(use_async_scheduling=False,
                                 drafter=SimpleNamespace(propose=propose, k=20, max_model_len=100),
                                 requests={rid: None, other: None},
                                 input_batch=SimpleNamespace(req_ids=[rid, other]),
                                 _bookkeeping_sync=lambda *a, **k: (None, None, None, [[1, 2, 999], [89]], None, [rid, other]))
        bridge = ReuseBridge(runner, "bank_suffix")
        self.assertEqual(bridge.register("missing", [7], 0)["status"], "error")
        bridge.register("a", [1, 2, 3, 4], 2)
        tokens = np.array([[151657, 1, 151657, 0], [2, 3, 4, 0]])
        # A tool marker in the prompt must not cause another request to receive a draft.
        drafts = bridge.propose(3, [[1], [1]], [3, 3], tokens)
        self.assertEqual(drafts, [[1, 2, 3], [89]])
        # Pending verification is retained even if D2 registers another prediction.
        bridge.register("a", [7, 8], 2)
        self.assertEqual(bridge.pending[rid], [1, 2, 3])
        bridge.bookkeeping(SimpleNamespace(scheduled_spec_decode_tokens={rid: [1, 2, 3]}))
        self.assertEqual(bridge.counts["accepted_draft_tokens"], 2)
        self.assertEqual(bridge.counts["accounting_errors"], 0)
        bridge.clear()
        self.assertFalse(bridge.pending or bridge.states or bridge.banks)

    def test_async_runner_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ReuseBridge(SimpleNamespace(use_async_scheduling=True))

    def test_replay_cannot_use_a_draft_that_arrives_after_generation(self):
        target = [151657, 11, 12, 13, 151658]
        events = [{"end_token": i, "received_s": float(i)} for i in range(1, 5)]
        late = simulate(target, 1, events, [(10.0, [11, 12, 13])], "single", 7)
        early = simulate(target, 1, events, [(0.0, [11, 12, 13])], "single", 7)
        self.assertEqual(late.get("accepted_tokens", 0), 0)
        self.assertEqual(early["accepted_tokens"], 3)

    def test_replay_counts_only_verified_prefix_before_correction(self):
        target = [151657, 1, 2, 99, 4, 5, 6, 151658]
        events = [{"end_token": i, "received_s": float(i)} for i in range(1, 8)]
        result = simulate(target, 1, events, [(0.0, [1, 2, 3, 4, 5, 6])], "single", 7)
        self.assertEqual(result["accepted_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
