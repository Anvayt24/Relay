"""Run the README's quickstart exactly as written, and check that it worked.

    python -m scripts.readme_quickstart        # needs Docker; CI runs it on every pull request

A quickstart that nobody runs goes stale: a renamed variable or a changed default and the first
thing a new user does fails. This takes the fenced block marked ```sh quickstart, runs it in
bash from the repository root in an isolated compose project (so it cannot touch a stack you
are using), and requires that its last command printed a chat completion.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```sh quickstart\n(.*?)^```", re.DOTALL | re.MULTILINE)


def extract_quickstart(readme: str) -> str:
    blocks = FENCE.findall(readme)
    if not blocks:
        raise ValueError("README.md has no fenced block marked ```sh quickstart")
    if len(blocks) > 1:
        raise ValueError("README.md has more than one ```sh quickstart block")
    return str(blocks[0])


def main() -> int:
    script = extract_quickstart((ROOT / "README.md").read_text(encoding="utf-8"))
    env = {**os.environ, "COMPOSE_PROJECT_NAME": "relayquickstart"}
    try:
        run = subprocess.run(
            ["bash", "-e", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
        )
        print(run.stdout)
        print(run.stderr, file=sys.stderr)
        if run.returncode != 0:
            print(f"quickstart failed (exit {run.returncode})", file=sys.stderr)
            return 1
        if '"choices"' not in run.stdout:
            print(
                "quickstart ran, but its last command did not return a completion", file=sys.stderr
            )
            return 1
        print("quickstart OK")
        return 0
    finally:
        subprocess.run(
            ["docker", "compose", "down", "-v"], cwd=ROOT, env=env, capture_output=True, check=False
        )


if __name__ == "__main__":
    raise SystemExit(main())
