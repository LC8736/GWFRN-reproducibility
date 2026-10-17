"""Export the synthetic source data, truth and partitions without fitting.

Fitting itself generates the identical arrays in memory. This optional export
makes the DGP-to-analysis workflow inspectable without storing 3,600 large
synthetic input files in the repository. It uses the same production generator.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
from data import generate_data
from settings import CONFIG, SCENARIOS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--run-start", type=int, default=1000)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    out = ROOT / "generated/simulation_data"
    out.mkdir(parents=True, exist_ok=True)
    for run_id in range(args.run_start, args.run_start+args.runs):
        for m in CONFIG["lattice_sizes"]:
            for scenario in CONFIG["scenarios"]:
                model, kappa = SCENARIOS[scenario]
                for error in CONFIG["errors"]:
                    d = generate_data(run_id, model, error, kappa=kappa, m=m, inner_folds=5)
                    np.savez_compressed(out / f"m{m}_{scenario}_{error}_r{run_id}.npz",
                                        coords=d.coords, x=d.x, latent_x=d.latent_x,
                                        z=d.z, y=d.y, conditional_mean=d.mu,
                                        eta=d.eta, zeta=d.zeta, basis_coefficients=d.bspline_coef,
                                        model_idx=d.model_idx, test_idx=d.test_idx,
                                        block_ids=d.block_ids,
                                        **{f"fit_{k}": a for k, (a,b) in enumerate(d.cv_folds)},
                                        **{f"validation_{k}": b for k, (a,b) in enumerate(d.cv_folds)})
        print(f"Exported replication {run_id}", flush=True)
