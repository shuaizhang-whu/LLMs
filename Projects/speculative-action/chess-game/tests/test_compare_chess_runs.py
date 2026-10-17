import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_chess_runs import compare_runs


class CompareChessRunsTests(unittest.TestCase):
    def _write(self, root: Path, mode: str, actions: list[str], fen: str) -> Path:
        path = root / mode / "metrics.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "execution_mode": mode,
                    "actions": actions,
                    "action_sequence_sha256": "same" if actions == ["[a2a3]"] else "different",
                    "final_fen": fen,
                    "steps_completed": len(actions),
                }
            ),
            encoding="utf-8",
        )
        return path.parent

    def test_equal_runs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequential = self._write(root, "sequential", ["[a2a3]"], "fen")
            speculative = self._write(root, "speculative", ["[a2a3]"], "fen")
            result = compare_runs(sequential, speculative)
            self.assertTrue(result["lossless_pass"])
            self.assertIsNone(result["first_mismatch"])

    def test_first_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequential = self._write(root, "sequential", ["[a2a3]"], "fen-a")
            speculative = self._write(root, "speculative", ["[h2h3]"], "fen-b")
            result = compare_runs(sequential, speculative)
            self.assertFalse(result["lossless_pass"])
            self.assertEqual(result["first_mismatch"]["step"], 0)


if __name__ == "__main__":
    unittest.main()
