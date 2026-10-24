"""CPU checks of real D3 proposal/binding/accounting, with no inference server."""
from types import SimpleNamespace
import unittest

from spork_core.draft_reuse import ProbeDraft
from spork_core.vllm029_d3 import D3Bridge
from scripts.replay_probe_reuse import ngram_proposal, replay
from scripts.check_probe_reuse_online import summarize


class DraftTests(unittest.TestCase):
    def test_reuse_requires_target_prefix_including_bonus_token(self):
        draft = ProbeDraft(list(range(50)), "prefix")
        self.assertEqual(draft.propose([], 20).tokens, list(range(20)))
        self.assertEqual(draft.propose(list(range(21)), 20).tokens, list(range(21, 41)))
        # A rejected draft's correction cannot be silently overwritten.
        self.assertIsNone(draft.propose(list(range(21)) + [999], 20))

    def test_single_injection_and_no_duplicate_same_step(self):
        draft = ProbeDraft([1, 2, 3, 4], "single")
        self.assertEqual(draft.propose([], 2).tokens, [1, 2])
        self.assertIsNone(draft.propose([1, 2, 3], 2))
        draft = ProbeDraft([1, 2, 3, 4], "prefix")
        draft.propose([], 2)
        self.assertIsNone(draft.propose([], 2))
        self.assertIsNone(draft.propose([1, 2, 3, 4], 2))

    def test_suffix_recovers_after_target_corrects_variable_length_value(self):
        draft = ProbeDraft([1, 2, 3, 4, 99, 10, 11, 12, 13, 14, 15], "suffix")
        p = draft.propose([1, 2, 3, 4, 777, 778, 10, 11, 12, 13], 20)
        self.assertEqual((p.tokens, p.source, p.matched_tokens), ([14, 15], "suffix", 4))

    def test_ambiguous_suffix_abstains(self):
        draft = ProbeDraft([1, 2, 3, 4, 8, 1, 2, 3, 4, 9], "suffix")
        self.assertIsNone(draft.propose([999, 1, 2, 3, 4], 20))

    def test_zero_budget_does_not_consume_single_injection(self):
        draft = ProbeDraft([1, 2], "single")
        self.assertIsNone(draft.propose([], 0))
        self.assertEqual(draft.propose([], 2).tokens, [1, 2])

    def test_ngram_keeps_earliest_match_and_overlapping_matches(self):
        self.assertEqual(ngram_proposal([1, 2, 9, 1, 2, 8, 1, 2], 1), [9])
        self.assertEqual(ngram_proposal([1, 1, 1, 1], 20), [1])

    def test_trace_replay_verifies_incorrect_proposals(self):
        for mode in ("single", "prefix", "suffix"):
            r = replay([90], [1, 2, 3, 4, 5], [1, 2, 99, 4, 5], mode, block=2)
            self.assertTrue(r["trace_reconstruction_exact"])
            self.assertEqual(r["first_probe_acceptance"], 2)


class Tokens(list):
    def tolist(self):
        return list(self)


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        row, cols = key
        return Tokens(self.rows[row][cols])


class Drafter:
    k = 20
    max_model_len = 1000

    def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None):
        return [[888] for _ in sampled_token_ids]


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.rid = "cmpl-main-0-engine"
        self.runner = SimpleNamespace(drafter=Drafter(), requests={self.rid: object()},
                                      input_batch=SimpleNamespace(req_ids=[self.rid, "other"]),
                                      samples=None)
        self.runner._bookkeeping_sync = lambda *_a, **_kw: (
            None, None, None, self.runner.samples, None, self.runner.input_batch.req_ids)
        self.bridge = D3Bridge(self.runner)

    def propose(self, body, limit=20):
        seq = [42, 151657] + body
        return self.bridge.propose(limit, [[1], [2]], [len(seq), 2], Matrix([seq, [7, 8]]))

    def verify(self, proposed, emitted):
        self.runner.samples = [emitted, [2]]
        return self.bridge.bookkeeping(SimpleNamespace(scheduled_spec_decode_tokens={self.rid: proposed}))

    def test_multiple_chunks_are_verified_and_bound_to_one_request(self):
        self.bridge.register("main", list(range(50)), 1, "prefix")
        first = self.propose([])
        self.assertEqual(first, [list(range(20)), [888]])
        # No second injection can overwrite an unverified first proposal.
        self.assertEqual(self.propose(list(range(21))), [[888], [888]])
        self.verify(list(range(20)), list(range(21)))
        second = self.propose(list(range(21)))
        self.assertEqual(second[0], list(range(21, 41)))
        self.verify(second[0], list(range(21, 42)))
        self.assertEqual(self.bridge.counts['verified_injections'], 2)
        self.assertEqual(self.bridge.counts['accepted_draft_tokens'], 40)
        self.assertEqual(self.bridge.counts['repeated_injections'], 1)
        self.assertEqual(self.bridge.counts['accounting_errors'], 0)

    def test_replacing_probe_keeps_pending_verification(self):
        self.bridge.register("main", [1, 2, 3], 1, "prefix")
        self.propose([])
        self.bridge.register("main", [1, 2, 9, 10], 1, "prefix")
        self.verify([1, 2, 3], [1, 2, 9])
        self.assertEqual(self.propose([1, 2, 9])[0], [10])
        self.assertEqual(self.bridge.counts['accepted_draft_tokens'], 2)

    def test_scheduler_may_shorten_proposal(self):
        self.bridge.register("main", [1, 2, 3, 4], 1, "prefix")
        self.propose([])
        self.verify([1, 2], [1, 2, 3])
        self.assertEqual(self.bridge.counts['accepted_draft_tokens'], 2)
        self.assertEqual(self.bridge.counts['accounting_errors'], 0)

    def test_no_proposal_after_tool_close_or_beyond_context_limit(self):
        self.bridge.register("main", [1, 2, 3], 1, "suffix")
        self.assertEqual(self.propose([151658]), [[888], [888]])
        self.runner.drafter.max_model_len = 2
        self.assertEqual(self.propose([]), [[888], [888]])

    def test_default_stays_single_and_clear_preserves_accounting(self):
        self.bridge.register("main", [1, 2, 3, 4], 1)
        self.propose([])
        self.verify([1, 2, 3, 4], [1, 2, 3, 4, 5])
        self.assertEqual(self.propose([1, 2, 3, 4, 5]), [[888], [888]])
        self.bridge.clear()
        self.assertFalse(self.bridge.states)
        self.assertFalse(self.bridge.pending)
        self.assertEqual(self.bridge.counts['dropped_injections'], 0)


class SmokeCoverageTests(unittest.TestCase):
    def test_matching_outputs_do_not_prove_reuse_was_exercised(self):
        modes = ["single", "prefix", "suffix"]
        records = [dict(group_id="retail:0", repeat=0, reuse_mode=mode, errors=[],
                        turns=[], action_reward=0, wall_s=1., reuse_counters={}) for mode in modes]
        manifest = dict(tasks=[dict(group_id="retail:0")], modes=modes, schedule=records)
        result = summarize(records, manifest)
        self.assertTrue(result["integrity_passed"])
        self.assertFalse(result["reuse_coverage_passed"])
        self.assertFalse(result["speedup_established"])
        records[1]["reuse_counters"] = dict(repeated_injections=2)
        records[2]["reuse_counters"] = dict(suffix_injections=1)
        self.assertTrue(summarize(records, manifest)["reuse_coverage_passed"])


if __name__ == "__main__":
    unittest.main()
