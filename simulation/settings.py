"""One explicit configuration shared by workers, models and reporting."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("GWFRN_CONFIG_PATH", ROOT / "config.json")).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
METHODS = ("GWR", "SGWR", "GNNWR", "SANNWR", "FNN", "GWFRN", "SGWFRN")
SCENARIOS = {"linear": ("model1", 1.0), "nonlinear_kappa0": ("model3", 0.0),
             "nonlinear_kappa1": ("model3", 1.0)}

def validate_config(config):
    if config["profile"] not in ("production", "smoke"):
        raise ValueError("profile must be production or smoke")
    if config["hidden_dim"] != 32 or config["first_hidden_dim"] != 64 or config["analysis_basis_dim"] != 15:
        raise ValueError("This implementation fixes M=15 and hidden widths 64/32")
    if config["relative_head_ridge"] <= 0 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Ridge, mini-batch size and learning rate must be positive")
    if not set(config["methods"]).issubset(METHODS) or not config["methods"]:
        raise ValueError("Unknown method; MGWR, GWANN and ablation variants are excluded")
    if not set(config["lattice_sizes"]).issubset({30, 40}) or not config["lattice_sizes"]:
        raise ValueError("Only m=30 and m=40 are supported")
    if not set(config["scenarios"]).issubset(SCENARIOS) or not config["scenarios"]:
        raise ValueError("Only the three main-experiment scenarios are supported")
    if not set(config["errors"]).issubset({"normal", "student_t", "bimodal"}) or not config["errors"]:
        raise ValueError("Unknown error distribution")
    for key in ("methods", "lattice_sizes", "scenarios", "errors", "bandwidths", "similarity_alphas", "smoothness"):
        if len(config[key]) != len(set(config[key])):
            raise ValueError(f"Duplicate entries in {key}")
    if not config["bandwidths"] or any(b <= 0 for b in config["bandwidths"]):
        raise ValueError("Bandwidth candidates must be positive")
    if not config["similarity_alphas"] or any(not 0 < a < 1 for a in config["similarity_alphas"]):
        raise ValueError("Similarity-guided alpha must be strictly between zero and one")
    if not config["smoothness"] or any(v <= 0 for v in config["smoothness"]):
        raise ValueError("The complete main methods retain positive smoothness penalties")
    interval = config["validation_interval"]
    if interval < 1 or len(config["stages"]) != 3:
        raise ValueError("Require a positive validation interval and three stages")
    previous_epochs = previous_folds = 0
    for stage in config["stages"]:
        if (not previous_epochs < stage["total_epochs"] or stage["total_epochs"] % interval
                or not previous_folds <= stage["folds"] <= 5 or stage["folds"] < 1
                or stage["retain"] < 1):
            raise ValueError("Invalid cumulative stage budgets/folds/retention")
        previous_epochs, previous_folds = stage["total_epochs"], stage["folds"]
    if config["stages"][-1]["folds"] != 5:
        raise ValueError("Final candidate and epoch must be selected using all five spatial folds")
    if config["replications"] < 1 or config["workers"] < 1 or config["run_start"] < 0:
        raise ValueError("Invalid replication/worker counts or starting seed index")

validate_config(CONFIG)
