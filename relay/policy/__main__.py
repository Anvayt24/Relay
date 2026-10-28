"""The policy CLI (PRD FR-11; ADR 0024). Needs RELAY_DATABASE_URL.

    python -m relay.policy create [--config relay.yaml] [--note "..."]   # bundle the YAML policy
    python -m relay.policy list                                          # bundles
    python -m relay.policy status                                        # pointer + recent events
    python -m relay.policy show 2026-10-28.1                             # one bundle (no artifacts)
    python -m relay.policy shadow 2026-10-28.2
    python -m relay.policy canary 2026-10-28.2 --pct 10
    python -m relay.policy promote                                       # a human's decision
    python -m relay.policy rollback [--reason "..."]
    python -m relay.policy activate 2026-10-28.1

Every change is one transaction (pointer + audit event); replicas apply it at their next poll
(5 s by default). ``create`` reads the router artifacts the YAML names and embeds them.
"""

import argparse
import asyncio
import getpass
import json
from pathlib import Path
from typing import Any

from relay.config.loader import DEFAULT_CONFIG, load_config
from relay.policy import control
from relay.policy.bundle import BundleError, from_config, next_version
from relay.policy.control import ControlError
from relay.policy.store import PolicyStore, PostgresPolicyStore, bundle_infos, describe
from relay.settings import Settings


async def run(args: argparse.Namespace, store: PolicyStore) -> int:
    actor = args.actor or getpass.getuser()
    try:
        match args.command:
            case "create":
                path = Path(args.config) if args.config else DEFAULT_CONFIG
                config = load_config(path)
                versions = [b.version for b in await store.bundles()]
                version = args.version or next_version(versions)
                bundle = from_config(config, path.parent, version=version, note=args.note)
                await store.create(bundle, created_by=actor)
                print(f"created bundle {version} (sha256 {bundle.content_sha256[:12]})")
            case "list":
                _print(bundle_infos(await store.bundles()))
            case "status":
                _print(
                    {
                        "pointer": describe(await store.pointer()),
                        "events": [
                            {"ts": e.ts.isoformat(), "action": e.action, "actor": e.actor,
                             "reason": e.reason}
                            for e in await store.events(10)
                        ],
                    }
                )  # fmt: skip
            case "show":
                found = await store.get(args.version)
                if found is None:
                    print(f"no bundle {args.version}")
                    return 1
                content = found.content()
                for route in content.values():  # artifacts are long; show their hash only
                    if route.get("router"):
                        route["router"]["artifact_json"] = "<embedded>"
                _print({"version": found.version, "note": found.note, "routes": content})
            case _:
                after = await store.change(
                    args.command, _transition(args), actor=actor, reason=args.reason or ""
                )
                _print({"pointer": describe(after)})
    except (ControlError, BundleError) as e:
        print(f"refused: {e}")
        return 1
    return 0


def _transition(args: argparse.Namespace) -> control.Transition:
    match args.command:
        case "shadow":
            return lambda p: control.shadow(p, args.version)
        case "canary":
            return lambda p: control.canary(p, args.version, args.pct)
        case "activate":
            return lambda p: control.activate(p, args.version)
        case "promote":
            return control.promote
        case _:
            return control.rollback


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m relay.policy", description=__doc__)
    p.add_argument("--actor", help="recorded on the audit event (default: your user name)")
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="bundle the policy in a YAML config")
    create.add_argument("--config", help="default: the built-in relay.yaml")
    create.add_argument("--version", help="default: the next YYYY-MM-DD.N")
    create.add_argument("--note", default="")
    sub.add_parser("list")
    sub.add_parser("status")
    show = sub.add_parser("show")
    show.add_argument("version")
    for name in ("shadow", "activate"):
        sub.add_parser(name).add_argument("version")
    canary = sub.add_parser("canary")
    canary.add_argument("version")
    canary.add_argument("--pct", type=int, required=True)
    sub.add_parser("promote")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--reason")
    for name in ("shadow", "canary", "promote", "activate"):
        sub.choices[name].add_argument("--reason")
    return p


async def _main(args: argparse.Namespace) -> int:
    dsn = Settings().database_url
    if not dsn:
        print("RELAY_DATABASE_URL is not set: bundles and the pointer live in Postgres")
        return 2
    store = PostgresPolicyStore(dsn)
    await store.start()
    try:
        return await run(args, store)
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_main(parser().parse_args(argv))))


if __name__ == "__main__":
    main()
