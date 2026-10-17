"""Single command interface for the complete manuscript data/model workflow.

This driver always launches fitting programs as separate Python processes.
Their different basis dimensions and module-level settings cannot contaminate
one another through Python's import cache. Paths are repository-relative.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    print("python " + " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, "-B", *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "preflight", "simulation", "realdata", "findings", "all"))
    parser.add_argument("--sim-workers", type=int, default=min(100, os.cpu_count() or 1))
    parser.add_argument("--real-workers", type=int, default=1)
    args = parser.parse_args()
    if args.sim_workers < 1 or not 1 <= args.real_workers <= 61:
        parser.error("Use positive worker counts; real-data Windows executor supports at most 61.")
    step = args.step
    if step in ("prepare", "all"):
        run("tools/prepare_data.py")
        run("realdata/run_annual_log.py", "--prepare-only")
    if step in ("preflight", "all"):
        run("tools/show_protocol.py")
        run("simulation/run_simulation.py", "--config", "simulation/config.json", "--preflight")
        run("realdata/run_annual_log.py", "--help")
    if step in ("simulation", "all"):
        # The production configuration contains every grid, law, model and
        # replication. No reduced budget or pilot profile is substituted.
        run("simulation/run_simulation.py", "--config", "simulation/config.json",
            "--output", "generated/simulation", "--workers", args.sim_workers,
            "--runs", 200, "--run-start", 1000)
        run("tools/build_findings.py", "--part", "simulation")
    if step in ("realdata", "all"):
        run("tools/prepare_data.py")
        run("realdata/run_annual_log.py", "--workers", args.real_workers)
        run("realdata/verify_results.py", "--require-all")
        run("tools/build_findings.py", "--part", "realdata")
        run("figures/plot_mean_absolute_errors.py")
    if step in ("findings", "all"):
        # Recalculate from newly fitted predictions, not the reference values.
        run("tools/build_findings.py", "--part", "all")
        run("figures/plot_mean_absolute_errors.py")


if __name__ == "__main__":
    main()
