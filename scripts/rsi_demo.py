#!/usr/bin/env python3
"""Two-run RSI demonstration — the no-training self-improvement claim, made measurable.

Runs the SAME op twice:
  1. COLD  — memory + kernel library for the op wiped first; the agent starts blind.
  2. WARM  — keeps everything COLD accumulated (strategy cards + best-kernel library +
             on-disk kernels); the seed agent is told a best kernel exists and pulls it
             from memory.

The headline observable is the speedup AT THE SEED PHASE (phase 0): the WARM run's very
first kernel should already match the COLD run's *final* kernel — i.e. run #2 starts where
run #1 finished, with no weight training, purely from accumulated memory.

Surgical: only this op's state is touched, so a prior full sweep's other-op memory is
preserved. A full backup of rsi/memory and rsi/kernels is taken first regardless.

Usage:  python3 scripts/rsi_demo.py [--op rmsnorm_residual] [--rounds 3] [--budget 5]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEM = ROOT / "rsi" / "memory"
KERNELS = ROOT / "rsi" / "kernels"
LIBRARY = MEM / "longterm" / "library"
STRATEGIES = MEM / "longterm" / "strategies.jsonl"
SHORTTERM = MEM / "shortterm"


def backup_state() -> Path:
    bdir = ROOT / "results" / "demo_backup"
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True, exist_ok=True)
    if MEM.exists():
        shutil.copytree(MEM, bdir / "memory")
    if KERNELS.exists():
        shutil.copytree(KERNELS, bdir / "kernels")
    return bdir


def wipe_op_state(op: str) -> None:
    """Remove only this op's memory + kernels — a true cold start for the op alone."""
    if STRATEGIES.exists():
        keep = [l for l in STRATEGIES.read_text().splitlines()
                if l.strip() and json.loads(l).get("op") != op]
        STRATEGIES.write_text(("\n".join(keep) + "\n") if keep else "")
    for ext in (".py", ".json"):
        p = LIBRARY / f"{op}{ext}"
        if p.exists():
            p.unlink()
    kdir = KERNELS / op
    if kdir.exists():
        shutil.rmtree(kdir)
    if SHORTTERM.exists():
        for f in SHORTTERM.glob(f"*_{op}.jsonl"):
            f.unlink()
    # also drop the op's leaderboard row, else a stale entry survives the wipe
    lb = ROOT / "results" / "leaderboard.json"
    if lb.exists():
        data = json.loads(lb.read_text())
        if data.pop(op, None) is not None:
            lb.write_text(json.dumps(data, indent=2))


def run_optimize(op: str, rounds: int, budget: float) -> dict:
    """Run one `rsi optimize` and return {run_id, cost, trajectory, final}."""
    proc = subprocess.run(
        ["rsi", "optimize", "--op", op, "--rounds", str(rounds), "--budget", str(budget)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    rid = re.search(r"run_id=(run_\d+)", out)
    run_id = rid.group(1) if rid else None
    cost_m = re.search(r"spent \$([0-9.]+)", out)
    cost = float(cost_m.group(1)) if cost_m else None

    traj = []
    if run_id:
        tf = SHORTTERM / f"{run_id}_{op}.jsonl"
        if tf.exists():
            traj = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]

    final = None
    lb = ROOT / "results" / "leaderboard.json"
    if lb.exists():
        final = json.loads(lb.read_text()).get(op)

    return {"run_id": run_id, "cost": cost, "trajectory": traj, "final": final,
            "ok": proc.returncode == 0, "stdout_tail": out[-1500:]}


def fmt_traj(traj: list) -> str:
    if not traj:
        return "    (no trajectory captured)"
    rows = []
    for t in traj:
        sp = t.get("best_speedup")
        sp_s = f"{sp:.3f}x" if isinstance(sp, (int, float)) else str(sp)
        rows.append(f"    phase {str(t.get('phase')):6s} -> best {sp_s:>9s}  (kernel {t.get('kernel_id')})")
    return "\n".join(rows)


def seed_speedup(traj: list):
    for t in traj:
        if t.get("phase") == "seed":
            return t.get("best_speedup")
    return None


def final_speedup(traj: list):
    vals = [t.get("best_speedup") for t in traj if isinstance(t.get("best_speedup"), (int, float))]
    return max(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-run RSI (cold vs warm) demonstration")
    ap.add_argument("--op", default="rmsnorm_residual")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=5.0)
    args = ap.parse_args()
    op = args.op

    print(f"=== RSI two-run demo: op={op} rounds={args.rounds} budget=${args.budget}/run ===\n")

    bdir = backup_state()
    print(f"[1/4] backed up memory+kernels -> {bdir}")

    wipe_op_state(op)
    print(f"[2/4] wiped '{op}' state (strategies + library + kernels + trajectories) — COLD start\n")

    print(f"[3/4] COLD run (empty memory for {op})…")
    t0 = time.time()
    cold = run_optimize(op, args.rounds, args.budget)
    print(f"      run_id={cold['run_id']}  cost=${cold['cost']}  ({time.time()-t0:.0f}s)")
    print(fmt_traj(cold["trajectory"]))
    if not cold["ok"]:
        print("      WARN: cold run returned non-zero\n" + cold["stdout_tail"])

    print(f"\n[4/4] WARM run (memory from COLD run now present)…")
    t0 = time.time()
    warm = run_optimize(op, args.rounds, args.budget)
    print(f"      run_id={warm['run_id']}  cost=${warm['cost']}  ({time.time()-t0:.0f}s)")
    print(fmt_traj(warm["trajectory"]))
    if not warm["ok"]:
        print("      WARN: warm run returned non-zero\n" + warm["stdout_tail"])

    cold_seed, warm_seed = seed_speedup(cold["trajectory"]), seed_speedup(warm["trajectory"])
    cold_fin, warm_fin = final_speedup(cold["trajectory"]), final_speedup(warm["trajectory"])

    def g(x):
        return f"{x:.3f}x" if isinstance(x, (int, float)) else str(x)

    print("\n" + "=" * 64)
    print("RSI RESULT  (memory-driven self-improvement, no training)")
    print("=" * 64)
    print(f"{'':22s}{'COLD (run 1)':>16s}{'WARM (run 2)':>16s}")
    print(f"{'seed-phase speedup':22s}{g(cold_seed):>16s}{g(warm_seed):>16s}")
    print(f"{'final speedup':22s}{g(cold_fin):>16s}{g(warm_fin):>16s}")
    print(f"{'cost (USD)':22s}{('$'+str(cold['cost'])):>16s}{('$'+str(warm['cost'])):>16s}")
    print("=" * 64)
    if isinstance(cold_fin, (int, float)) and isinstance(warm_seed, (int, float)):
        if warm_seed >= cold_fin * 0.95:
            print(f"✓ WARM run's FIRST kernel ({g(warm_seed)}) already matches COLD run's "
                  f"BEST ({g(cold_fin)}).\n  Run #2 started where run #1 finished — purely from memory.")
        else:
            print(f"… WARM seed {g(warm_seed)} vs COLD final {g(cold_fin)} — see trajectories above.")

    report = ROOT / "results" / "rsi_demo_report.json"
    report.write_text(json.dumps({"op": op, "cold": cold, "warm": warm}, indent=2, default=str))
    print(f"\nreport -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
