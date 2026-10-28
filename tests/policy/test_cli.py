"""The policy CLI against the in-memory store (the Postgres store is the integration test)."""

import json

import pytest

from relay.policy import __main__ as cli
from relay.policy.store import MemoryPolicyStore

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]


async def run(store: MemoryPolicyStore, *argv: str) -> int:
    return await cli.run(cli.parser().parse_args(["--actor", "tester", *argv]), store)


async def test_create_then_canary_promote_and_rollback(capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryPolicyStore()
    assert await run(store, "create", "--version", "2026-10-28.1", "--note", "first") == 0
    assert await run(store, "create", "--note", "second") == 0  # next free version
    versions = [b.version for b in await store.bundles()]
    assert versions[0] == "2026-10-28.1" and len(versions) == 2
    second = versions[1]

    assert await run(store, "activate", "2026-10-28.1") == 0
    assert await run(store, "canary", second, "--pct", "10") == 0
    p = await store.pointer()
    assert (p.candidate_version, p.canary_pct) == (second, 10)
    assert await run(store, "promote", "--reason", "card safe") == 0
    assert (await store.pointer()).active_version == second
    assert await run(store, "rollback") == 0
    assert (await store.pointer()).active_version == "2026-10-28.1"
    events = await store.events()
    assert [e.action for e in reversed(events)] == ["activate", "canary", "promote", "rollback"]
    assert {e.actor for e in events} == {"tester"}

    capsys.readouterr()
    assert await run(store, "status") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["pointer"]["active_version"] == "2026-10-28.1"
    assert await run(store, "list") == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert await run(store, "show", "2026-10-28.1") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["routes"]["default"]["router"]["artifact_json"] == "<embedded>"


async def test_refusals_exit_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryPolicyStore()
    assert await run(store, "promote") == 1  # nothing to promote
    assert await run(store, "activate", "2026-01-01.1") == 1  # no such bundle
    assert await run(store, "show", "2026-01-01.1") == 1
    assert await run(store, "create", "--version", "2026-10-28.1") == 0
    assert await run(store, "create", "--version", "2026-10-28.1") == 1  # bundles are immutable
    assert "refused" in capsys.readouterr().out


def test_the_cli_needs_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: relay_settings())
    with pytest.raises(SystemExit) as exit_:
        cli.main(["status"])
    assert exit_.value.code == 2
