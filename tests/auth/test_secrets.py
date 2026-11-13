"""Generated secrets: a first run needs no hand-made pepper or admin token.

The values live in RELAY_DATA_DIR (a volume), owner-only. An environment value always wins,
and a file that cannot be trusted is an error, never a silent new secret.
"""

import os
import threading
from pathlib import Path

import pytest

from relay.auth.secrets import load_or_create_secret
from relay.settings import Settings


def settings(tmp_path: Path, **values: object) -> Settings:
    return Settings(_env_file=None, data_dir=str(tmp_path), **values)  # type: ignore[call-arg]


def test_a_secret_is_created_once_and_reused(tmp_path: Path) -> None:
    first, created = load_or_create_secret(tmp_path, "key_pepper")
    second, created_again = load_or_create_secret(tmp_path, "key_pepper")
    assert created and not created_again
    assert first == second and len(first) >= 32
    assert load_or_create_secret(tmp_path, "admin_token")[0] != first  # one per name


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_file_is_readable_by_its_owner_only(tmp_path: Path) -> None:
    load_or_create_secret(tmp_path, "key_pepper")
    assert (tmp_path / "key_pepper").stat().st_mode & 0o777 == 0o600


def test_replicas_starting_together_all_get_the_same_secret(tmp_path: Path) -> None:
    results: list[str] = []

    def start() -> None:
        results.append(load_or_create_secret(tmp_path, "key_pepper")[0])

    threads = [threading.Thread(target=start) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1 and len(results) == 12


def test_an_empty_file_is_refused_not_replaced(tmp_path: Path) -> None:
    # An empty pepper would hash every key the same way; a regenerated one would silently
    # invalidate every key already issued. Neither is a start the operator asked for.
    (tmp_path / "key_pepper").write_text("  \n")
    with pytest.raises(ValueError, match="key_pepper"):
        load_or_create_secret(tmp_path, "key_pepper")


def test_settings_take_the_pepper_from_the_data_dir(tmp_path: Path) -> None:
    first = settings(tmp_path)
    assert first.key_pepper is not None and first.key_pepper_generated
    again = settings(tmp_path)
    assert again.key_pepper is not None and not again.key_pepper_generated
    assert again.key_pepper.get_secret_value() == first.key_pepper.get_secret_value()


def test_an_environment_pepper_wins_and_writes_nothing(tmp_path: Path) -> None:
    s = settings(tmp_path, key_pepper="from-the-environment")
    assert s.key_pepper is not None and s.key_pepper.get_secret_value() == "from-the-environment"
    assert not s.key_pepper_generated and not (tmp_path / "key_pepper").exists()


def test_without_a_data_dir_nothing_is_generated() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.key_pepper is None and not s.key_pepper_generated


def test_the_admin_token_is_generated_only_when_the_admin_api_is_on(tmp_path: Path) -> None:
    off = settings(tmp_path)
    assert off.admin_token is None and not (tmp_path / "admin_token").exists()
    on = settings(tmp_path, admin_api="on")
    assert on.admin_token is not None and (tmp_path / "admin_token").exists()
    assert on.admin_token.get_secret_value() != on.key_pepper.get_secret_value()  # type: ignore[union-attr]
    explicit = settings(tmp_path, admin_api="on", admin_token="mine")
    assert explicit.admin_token is not None and explicit.admin_token.get_secret_value() == "mine"
