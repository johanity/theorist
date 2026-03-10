"""CLI interface.

Usage:
    theorist brain              -- show accumulated knowledge
    theorist brain --reset      -- clear the brain
    theorist brain --export     -- export theory as JSON
    theorist history            -- show recent experiments
    theorist history -n 20      -- show last 20
    theorist run FILE           -- run a decorated experiment file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .brain import Brain


def cmd_brain(args: argparse.Namespace) -> None:
    brain = Brain(path=args.brain_path)
    if args.reset:
        brain.reset()
        print("Brain reset.")
        return
    if args.export:
        print(json.dumps(brain.export_anonymized(), indent=2))
        return
    print(brain.summary())


def cmd_history(args: argparse.Namespace) -> None:
    brain = Brain(path=args.brain_path)
    if not brain.experiments_file.exists():
        print("No experiments yet.")
        return

    lines = brain.experiments_file.read_text().strip().split("\n")
    n = args.n or len(lines)
    for line in lines[-n:]:
        exp = json.loads(line)
        domain = exp.get("domain", "?")
        eid = exp.get("exp_id", "?")
        actual = exp.get("actual", 0)
        surprise = exp.get("surprise", 0)
        status = exp.get("status", "?")
        desc = exp.get("description", "")[:40]
        print(f"  [{domain}] exp {eid:>3}: {actual:>10.4f}  "
              f"surp={surprise:.3f}  {status:>7}  {desc}")


def cmd_run(args: argparse.Namespace) -> None:
    filepath = args.file
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location("user_experiment", filepath)
    if spec is None or spec.loader is None:
        print(f"Cannot load: {filepath}")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(filepath).parent))
    spec.loader.exec_module(mod)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="theorist",
        description="Theorist -- prediction-error learning for optimization",
    )
    parser.add_argument("--brain-path", default="~/.theorist",
                        help="Path to brain storage")

    sub = parser.add_subparsers(dest="command")

    p_brain = sub.add_parser("brain", help="Show brain summary")
    p_brain.add_argument("--reset", action="store_true", help="Reset the brain")
    p_brain.add_argument("--export", action="store_true", help="Export theory as JSON")

    p_hist = sub.add_parser("history", help="Show experiment history")
    p_hist.add_argument("-n", type=int, default=None, help="Number of recent experiments")

    p_run = sub.add_parser("run", help="Run an experiment file")
    p_run.add_argument("file", help="Path to Python file with @experiment decorator")

    args = parser.parse_args()

    commands = {
        "brain": cmd_brain,
        "history": cmd_history,
        "run": cmd_run,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
