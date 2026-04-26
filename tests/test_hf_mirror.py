import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class HuggingFaceMirrorTests(unittest.TestCase):
    def _run_endpoint_probe(self, env_overrides=None):
        env = os.environ.copy()
        env.pop("HF_ENDPOINT", None)
        env.update(env_overrides or {})

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import app; "
                    "import os; "
                    "from huggingface_hub import constants; "
                    "print(os.environ.get('HF_ENDPOINT')); "
                    "print(constants.ENDPOINT)"
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def test_app_defaults_hf_endpoint_to_domestic_mirror(self):
        lines = self._run_endpoint_probe()

        self.assertEqual(lines[-2:], ["https://hf-mirror.com", "https://hf-mirror.com"])

    def test_app_preserves_preconfigured_hf_endpoint(self):
        lines = self._run_endpoint_probe({"HF_ENDPOINT": "https://huggingface.co"})

        self.assertEqual(lines[-2:], ["https://huggingface.co", "https://huggingface.co"])


if __name__ == "__main__":
    unittest.main()
