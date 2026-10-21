"""The commit hook runs with only the Python standard library."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "check_jsonl.py"


class CheckJsonlTest(unittest.TestCase):
    def test_cli_checks_multiple_files_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.jsonl"
            bad = Path(directory) / "bad.jsonl"
            good.write_text('{"state": "Zażółć"}\n\n{"questions": {}}\n', encoding="utf-8")
            bad.write_text('{}\n{"broken":\n', encoding="utf-8")
            before = {path: path.read_bytes() for path in (good, bad)}
            valid = subprocess.run([sys.executable, str(HOOK), str(good)], capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            invalid = subprocess.run(
                [sys.executable, str(HOOK), str(bad), str(good)], capture_output=True, text=True,
            )
            self.assertEqual(invalid.returncode, 1)
            self.assertIn(f"{bad}:2:", invalid.stdout)
            self.assertEqual(before, {path: path.read_bytes() for path in before})


if __name__ == "__main__":
    unittest.main()
