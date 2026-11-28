import json
from pathlib import Path
import tempfile
import unittest

from spork_core.eval_artifacts import archive_unsealed, freeze_manifest, load_sealed, seal_record


class ArtifactTests(unittest.TestCase):
    def test_resume_rejects_changed_policy_without_rewriting_original_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = {"created_at": "a", "tasks": ["train:1"], "temperature": .6}
            freeze_manifest(root, manifest)
            original = (Path(root) / "manifest.json").read_bytes()
            self.assertEqual(freeze_manifest(root, {**manifest, "created_at": "b"}, resume=True), manifest)
            with self.assertRaisesRegex(ValueError, "change the experiment"):
                freeze_manifest(root, {**manifest, "temperature": 0}, resume=True)
            self.assertEqual((Path(root) / "manifest.json").read_bytes(), original)

    def test_failed_runs_are_sealed_and_tampered_records_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "record.json"
            failure = {"errors": ["Actor length limit"], "wall_s": 10}
            seal_record(path, failure)
            self.assertEqual(load_sealed(path), failure)
            with self.assertRaises(FileExistsError):
                seal_record(path, {"errors": [], "wall_s": 5})
            path.write_text(json.dumps({"errors": [], "wall_s": 5}))
            with self.assertRaisesRegex(ValueError, "modified"):
                load_sealed(path)

    def test_unsealed_attempt_is_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "record.json"
            path.write_text('{"partial": true}')
            self.assertIsNone(load_sealed(path))
            backup = archive_unsealed(path)
            self.assertEqual(backup.read_text(), '{"partial": true}')
            self.assertFalse(path.exists())
            seal_record(path, {"errors": []})
            with self.assertRaisesRegex(ValueError, "sealed"):
                archive_unsealed(path)
