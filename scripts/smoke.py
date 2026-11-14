"""An end-to-end check of the self-hosted stack, with no provider keys (make smoke).

    python -m scripts.smoke        # needs Docker

Brings up the default stack (gateway, Postgres, Redis) in an isolated compose project, then:
the first API key is printed once, a request without it is refused, a request with it is
answered by the built-in simulator, the request is recorded in Postgres, and metrics are not
served on the public port. Everything is torn down afterwards, volumes included.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PORT = os.environ.get("RELAY_SMOKE_PORT", "18000")
BASE = f"http://127.0.0.1:{PORT}"
ENV = {**os.environ, "COMPOSE_PROJECT_NAME": "relaysmoke", "RELAY_PORT": PORT}


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, env=ENV, capture_output=True, text=True, check=check
    )


def http(
    path: str, *, key: str | None = None, body: dict[str, Any] | None = None
) -> tuple[int, str]:
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def first_key(timeout_s: float = 60) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = re.search(r"\brk_[A-Za-z0-9_-]{20,}", compose("logs", "api").stdout)
        if found:
            return found.group(0)
        time.sleep(1)
    raise AssertionError("the first API key was not printed in the gateway's logs")


def expect(condition: bool, what: str) -> None:
    print(("ok   " if condition else "FAIL ") + what)
    if not condition:
        raise SystemExit(1)


def main() -> int:
    try:
        compose("up", "-d", "--build", "--wait")
        key = first_key()
        expect(True, "the first API key was printed")
        status, _ = http("/v1/models")
        expect(status == 401, f"a request without a key is refused (got {status})")
        status, text = http(
            "/v1/chat/completions",
            key=key,
            body={"model": "relay/cheap", "messages": [{"role": "user", "content": "smoke"}]},
        )
        expect(
            status == 200 and "choices" in text, f"a request with the key is answered ({status})"
        )
        time.sleep(2)  # the request log is written in batches, every 0.5 s
        rows = compose(
            "exec", "-T", "postgres", "psql", "-U", "relay", "-tAc", "select count(*) from requests"
        ).stdout.strip()
        expect(rows.isdigit() and int(rows) >= 1, f"the request was recorded in Postgres ({rows})")
        status, _ = http("/metrics")
        expect(status == 404, f"metrics are not on the public port (got {status})")
        print("smoke OK")
        return 0
    finally:
        if "--keep" not in sys.argv:
            compose("down", "-v", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
