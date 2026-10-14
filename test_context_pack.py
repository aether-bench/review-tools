"""Regression tests for context_pack.py: each builds a two-commit repository, runs the tool on it and reads the pack.
Run with `python3 -m unittest test_context_pack`. Requires git and python3 on PATH."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parent / "context_pack.py"
GIT_IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


class Repo:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.dir.name)
        self.git("init", "-q", "-b", "main")

    def git(self, *args):
        return subprocess.run(["git", *GIT_IDENTITY, *args], cwd=self.path, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, files: dict, message: str) -> str:
        for name, body in files.items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if body is None:
                target.unlink()
            else:
                target.write_text(body)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def pack(self, base: str, env: dict = None, expect_failure: bool = False) -> str:
        out = self.path.parent / f"{self.path.name}-pack.md"
        run_env = {**os.environ, **(env or {})}
        r = subprocess.run([sys.executable, "-B", str(TOOL), "--base", base, "--head", "HEAD", "--out", str(out)],
                           cwd=self.path, capture_output=True, text=True, env=run_env)
        if expect_failure:
            assert r.returncode != 0, "expected the tool to fail"
            return r.stderr
        assert r.returncode == 0, r.stderr
        return out.read_text()


def two_commit(base_files: dict, head_files: dict):
    repo = Repo()
    base = repo.commit(base_files, "base")
    repo.commit(head_files, "head")
    return repo, base


class ContextPackTests(unittest.TestCase):
    def test_hunk_inside_its_function_and_skeleton(self):
        repo, base = two_commit(
            {"billing.py": "def total(items):\n    amount = 0\n    for item in items:\n        amount += item.price\n    return amount\n"},
            {"billing.py": "def total(items):\n    amount = 0\n    for item in items:\n        amount += item.price * item.quantity_pct\n    return amount\n"})
        pack = repo.pack(base)
        self.assertIn("## Changed files", pack)
        self.assertIn("- `billing.py` (modified; +1 -1; 1 hunks)", pack)
        self.assertIn("amount += item.price * item.quantity_pct", pack)
        self.assertIn("Skeleton of `billing.py`", pack)

    def test_non_ascii_path_is_read_as_git_names_it(self):
        repo, base = two_commit({"café.py": "def brew():\n    return 1\n"}, {"café.py": "def brew():\n    return 2\n"})
        pack = repo.pack(base)
        self.assertIn("- `café.py` (modified; +1 -1; 1 hunks)", pack)
        self.assertIn("Skeleton of `café.py`", pack)

    def test_payload_lines_that_look_like_file_headers_are_content(self):
        repo, base = two_commit(
            {"q.sql": "-- name: one\nselect 1;\n-- name: two\nselect 2;\n", "c.c": "int f(void) {\n  int counter = 0;\n  return counter;\n}\n"},
            {"q.sql": "select 1;\n-- name: two\nselect 2;\n", "c.c": "int f(void) {\n  int counter = 0;\n++ counter;\n  return counter;\n}\n"})
        pack = repo.pack(base)
        self.assertIn("-   -1| -- name: one", pack)
        self.assertIn("    2| -- name: two", pack)
        self.assertIn("+    3| ++ counter;", pack)
        self.assertIn("- `q.sql` (modified; +0 -1; 1 hunks)", pack)

    def test_a_call_is_not_a_definition(self):
        repo, base = two_commit(
            {"util.py": "def calculate_total(items):\n    return sum(i.price for i in items)\n", "app.py": "def checkout(items):\n    return 0\n"},
            {"util.py": "def calculate_total(items):\n    return sum(i.price for i in items)\n", "app.py": "def checkout(items):\n    return calculate_total(items)\n"})
        pack = repo.pack(base)
        self.assertIn("### `calculate_total` (called at app.py:2): 1 definition", pack)
        self.assertIn("1| def calculate_total(items):", pack)
        self.assertNotIn("return calculate_total(items):", pack.split("## Definitions of functions the added lines call", 1)[1])

    def test_enclosing_function_found_inside_the_hunk(self):
        repo, base = two_commit(
            {"math.py": "def total(items):\n    amount = 0\n    return amount\n", "main.py": "from math import total\n\nvalue = total([])\n"},
            {"math.py": "def total(items):\n    amount = 1\n    return amount\n", "main.py": "from math import total\n\nvalue = total([])\n"})
        pack = repo.pack(base)
        self.assertIn("## References to changed and removed functions", pack)
        self.assertIn("- `total` (changed function at math.py:1): 2 references", pack)
        self.assertIn("    - main.py:3: value = total([])", pack)

    def test_renamed_literal_still_lists_its_stale_consumer(self):
        repo, base = two_commit(
            {"consts.py": 'EVENT = "workspace_ready"\n', "consumer.py": 'def on(event):\n    return event == "workspace_ready"\n'},
            {"consts.py": 'EVENT = "workspace_ready_v2"\n', "consumer.py": 'def on(event):\n    return event == "workspace_ready"\n'})
        pack = repo.pack(base)
        removed = pack.split("## Literals the change removes, with where they are still used", 1)[1]
        self.assertIn("- `workspace_ready` at consts.py:-1: still referenced elsewhere: 1", removed)
        self.assertIn("consumer.py:2", removed)

    def test_forced_grep_colour_does_not_hide_occurrences(self):
        repo, base = two_commit(
            {"consts.py": 'EVENT = "workspace_ready"\n', "consumer.py": 'def on(event):\n    return event == "workspace_ready"\n'},
            {"consts.py": 'EVENT = "workspace_ready_v2"\n', "consumer.py": 'def on(event):\n    return event == "workspace_ready"\n'})
        pack = repo.pack(base, env={"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "color.grep", "GIT_CONFIG_VALUE_0": "always"})
        self.assertIn("consumer.py:2", pack.split("## Literals the change removes", 1)[1])

    def test_failed_run_leaves_no_temporary_files(self):
        repo, base = two_commit({"a.py": "x = 1\n"}, {"a.py": "x = 2\n"})
        with tempfile.TemporaryDirectory() as tmp:
            stderr = repo.pack("no-such-ref", env={"TMPDIR": tmp, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "diff.noprefix", "GIT_CONFIG_VALUE_0": "true"}, expect_failure=True)
            self.assertIn("failed", stderr)
            self.assertEqual(os.listdir(tmp), [])
        pack = repo.pack(base, env={"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "diff.noprefix", "GIT_CONFIG_VALUE_0": "true"})
        self.assertIn("- `a.py` (modified; +1 -1; 1 hunks)", pack)


if __name__ == "__main__":
    unittest.main()
