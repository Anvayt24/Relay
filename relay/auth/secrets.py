"""Secrets Relay generates for itself on a first run.

A self-hosted gateway should start without the operator inventing a pepper. When
``RELAY_DATA_DIR`` names a directory (a volume, in Docker), a secret that is not set in the
environment is created there once, owner-only, and reused on every later start.

Two rules keep this safe:
- **An environment value always wins.** An operator who manages secrets elsewhere is never
  second-guessed, and nothing is written.
- **A file that cannot be trusted is an error.** An empty pepper would hash every key the same
  way, and quietly replacing a lost one would invalidate every key already issued.

Creation is atomic: the value is written to a private temp file and then *linked* to its final
name, which fails if the name exists. Replicas starting together all end up reading the one
value that won, never a half-written file.
"""

import os
import secrets
import tempfile
from pathlib import Path


def load_or_create_secret(directory: str | Path, name: str) -> tuple[str, bool]:
    """(value, created). `created` is True only for the process that wrote the file."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    final = base / name
    if final.exists():
        return _read(final, name), False

    value = secrets.token_urlsafe(48)
    fd, temp = tempfile.mkstemp(dir=base, prefix=f".{name}.")  # created 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value + "\n")
        try:
            os.link(temp, final)  # atomic, and refuses to replace an existing file
        except FileExistsError:
            return _read(final, name), False  # another replica won the race
        return value, True
    finally:
        os.unlink(temp)


def _read(path: Path, name: str) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(
            f"{path} is empty. Relay will not invent a new {name}: it would invalidate "
            "everything made with the old one. Restore the file, or set the value in the "
            "environment."
        )
    return value
