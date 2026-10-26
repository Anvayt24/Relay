"""Run the simulator as an HTTP server: ``python -m relay.simulator --scenario realistic``."""

import argparse

import uvicorn

from relay.simulator.engine import Simulator
from relay.simulator.loader import load_scenario
from relay.simulator.scenario import PRESETS
from relay.simulator.server import create_sim_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m relay.simulator",
        description="Serve a simulated LLM provider (OpenAI and Anthropic wire formats).",
    )
    parser.add_argument("--scenario", default="realistic", choices=sorted(PRESETS))
    parser.add_argument("--scenario-file", help="a YAML scenario; overrides --scenario")
    parser.add_argument(
        "--control",
        action="store_true",
        help="mount /sim/control, /sim/clear and /sim/reset (localhost chaos runs only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario_file) if args.scenario_file else PRESETS[args.scenario]
    simulator = Simulator.from_scenario(scenario, seed=args.seed)
    uvicorn.run(
        create_sim_app(simulator, control=args.control),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
