"""Command-line entrypoint:  `rsi <command>`

  rsi optimize [--op NAME|all] [--rounds N] [--passes P] [--autonomous] [--budget USD]
  rsi leaderboard
  rsi ops
"""
from __future__ import annotations

import argparse
import asyncio
import json

from . import config
from .bench.ops import DEFAULT_OP_ORDER


def _cmd_leaderboard(_args) -> None:
    if config.LEADERBOARD.exists():
        lb = json.loads(config.LEADERBOARD.read_text())
        print(f"Leaderboard ({config.DIMS.name} on {config.HW.name}):")
        if not lb:
            print("  (empty)")
        for op, rec in sorted(lb.items(), key=lambda kv: kv[1].get("speedup", 0), reverse=True):
            flag = "✓" if rec.get("speedup", 0) >= config.TARGET_SPEEDUP else " "
            print(f"  [{flag}] {op:24s} {rec.get('speedup', 0):.3f}x  (kernel {rec.get('kernel_id')})")
    else:
        print("No leaderboard yet — run `rsi optimize` first.")


def _cmd_ops(_args) -> None:
    print(f"Target ops for {config.DIMS.name} (priority order):")
    for name in DEFAULT_OP_ORDER:
        print(f"  - {name}")
    print("\nTip: `rsi optimize --op <name>` or `--op all`.")


def _cmd_auth(_args) -> None:
    st = config.enforce_subscription_auth(strict=False)
    print("Auth mode for rsi-spawned agents:")
    if st["subscription_present"]:
        print(f"  ✓ Claude SUBSCRIPTION ({st['subscription_type']}) — ~/.claude OAuth token")
    else:
        print("  ✗ no subscription OAuth token at ~/.claude/.credentials.json")
    if st["removed"]:
        print(f"  ⚠ removed API-billing env vars from this process: {', '.join(st['removed'])}")
    else:
        print("  ✓ no API key / alternate-provider env vars set — nothing to strip")
    print("  → billing: your Claude subscription (no metered API credits)."
          if st["subscription_present"] and "ANTHROPIC_API_KEY" not in st["removed"]
          else "  → check the warnings above.")


def _cmd_optimize(args) -> None:
    # Hard guarantee: subscription billing only, never an API key.
    auth = config.enforce_subscription_auth()
    from .orchestrator import Orchestrator

    ops = DEFAULT_OP_ORDER if args.op == "all" else [args.op]
    orch = Orchestrator(per_run_usd=args.budget)
    print(f"[rsi] auth=subscription({auth['subscription_type']})  run_id={orch.run_id}  ops={ops}  "
          f"rounds={args.rounds}  passes={args.passes}  budget≈${orch.budget_cap}(notional)  "
          f"autonomous={args.autonomous}")

    async def _go():
        if args.autonomous:
            return await orch.run_autonomous(ops)
        return await orch.optimize_all(ops, rounds=args.rounds, passes=args.passes)

    result = asyncio.run(_go())
    print("\n[rsi] done. spent ${:.4f}".format(orch.spent))
    print(json.dumps(result, indent=2, default=str))
    print()
    _cmd_leaderboard(args)


def main() -> None:
    p = argparse.ArgumentParser(prog="rsi", description="Recursive kernel-optimization agent framework")
    sub = p.add_subparsers(dest="command", required=True)

    po = sub.add_parser("optimize", help="optimize one op or all ops")
    po.add_argument("--op", default="all", help="op name or 'all' (default: all)")
    po.add_argument("--rounds", type=int, default=config.BUDGET.rounds, help="refinement phases per op")
    po.add_argument("--passes", type=int, default=1, help="outer re-attack passes over the op set")
    po.add_argument("--budget", type=float, default=config.BUDGET.per_run_usd, help="USD cap for the run")
    po.add_argument("--autonomous", action="store_true",
                    help="single Opus orchestrator delegating via Task (pure hierarchical mode)")
    po.set_defaults(func=_cmd_optimize)

    sub.add_parser("leaderboard", help="print the current best kernel per op").set_defaults(func=_cmd_leaderboard)
    sub.add_parser("ops", help="list target ops").set_defaults(func=_cmd_ops)
    sub.add_parser("auth", help="show billing/auth mode (subscription vs API)").set_defaults(func=_cmd_auth)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
