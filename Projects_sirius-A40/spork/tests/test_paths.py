"""Local relocation checks; no model imports, services, or network access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PathTests(unittest.TestCase):
    def environment(self, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith('SPORK_')}
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        env.update(overrides)
        return env

    def python_paths(self, **overrides):
        code = '''
import json
from spork_core.paths import REPO_ROOT, MODEL_ROOT, TAU2_ROOT, resolve_artifact_path
print(json.dumps([str(p) for p in [REPO_ROOT, MODEL_ROOT, TAU2_ROOT,
 resolve_artifact_path('/old/host/LLMs/Projects/spork/controller_runs/example'),
 resolve_artifact_path('/old/host/LLMs/Models/bert-tiny'),
 resolve_artifact_path('results/example'), resolve_artifact_path('/explicit/assets/file')]]))
'''
        return json.loads(subprocess.check_output(
            [sys.executable, '-c', code], cwd='/tmp', env=self.environment(**overrides), text=True))

    def test_copied_metadata_uses_local_checkout_from_another_cwd(self):
        repo, models, tau2, checkpoint, encoder, result, explicit = self.python_paths()
        self.assertEqual(Path(repo), ROOT)
        self.assertEqual(Path(models), ROOT.parent.parent / 'Models')
        self.assertEqual(Path(tau2), ROOT / 'tau2-bench')
        self.assertEqual(Path(checkpoint), ROOT / 'controller_runs/example')
        self.assertEqual(Path(encoder), Path(models) / 'bert-tiny')
        self.assertEqual(Path(result), ROOT / 'results/example')
        self.assertEqual(explicit, '/explicit/assets/file')

    def test_relative_overrides_are_project_relative(self):
        values = self.python_paths(SPORK_MODEL_ROOT='weights', SPORK_TAU2_ROOT='deps/tau2')
        self.assertEqual(Path(values[1]), ROOT / 'weights')
        self.assertEqual(Path(values[2]), ROOT / 'deps/tau2')

    def test_standalone_checkout_and_shell_agree_without_changing_cwd(self):
        with tempfile.TemporaryDirectory(prefix='spork paths ') as directory:
            repo = Path(directory) / 'standalone'
            repo.mkdir()
            env = self.environment(SPORK_REPO_ROOT=str(repo), SPORK_PYTHON=sys.executable)
            before = self.python_paths(SPORK_REPO_ROOT=str(repo))
            self.assertEqual(Path(before[1]), repo / 'models')
            shell = 'source "$1"; printf "%s\\n" "$PWD" "$SPORK_REPO_ROOT" "$SPORK_MODEL_ROOT" "$SPORK_PYTHON"'
            lines = subprocess.check_output(['bash', '-euc', shell, 'bash', str(ROOT / 'scripts/spork_env.sh')],
                                            cwd='/tmp', env=env, text=True).splitlines()
            self.assertEqual(lines, ['/tmp', str(repo), str(repo / 'models'), sys.executable])

    def test_shell_relative_model_override_matches_python(self):
        shell = 'source "$1"; printf "%s\\n" "$SPORK_MODEL_ROOT"'
        actual = subprocess.check_output(['bash', '-euc', shell, 'bash', str(ROOT / 'scripts/spork_env.sh')],
                cwd='/tmp', env=self.environment(SPORK_MODEL_ROOT='weights', SPORK_PYTHON=sys.executable), text=True).strip()
        self.assertEqual(actual, self.python_paths(SPORK_MODEL_ROOT='weights')[1])


if __name__ == '__main__':
    unittest.main()
