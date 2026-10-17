"""Legacy GWR and the public SGWR selector, independent of proposed grids."""
from __future__ import annotations
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent


def features(data, indices):
    raw = np.column_stack((data.bspline_coef, data.z))
    mean, scale = raw[indices].mean(0), raw[indices].std(0)
    scale[scale < 1e-12] = 1.0
    return (raw - mean) / scale


def fit_gwr(data, output):
    """Use the fixed distance-quantile spatial-CV GWR implementation."""
    from legacy_gwr import fit_gwr as quantile_fit_gwr
    result = quantile_fit_gwr(data, pilot=False)
    result["selected"].update(
        criterion="pooled spatial-block CV MSE",
        bandwidth_quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
        search="distance-quantile candidates",
    )
    result["protocol"] = dict(
        proposed_candidate_grid_used=False,
        source="distance-quantile GWR adapter",
        gaussian_fixed_kernel=True,
        validation_gram_safeguard=1e-9,
        distance_scale_sample="outer modelling locations",
    )
    Path(output, "legacy_gwr_selected.json").write_text(
        json.dumps(result["selected"], indent=2), encoding="utf-8"
    )
    return result


def official_alpha_class():
    source = ROOT / "upstream/FastSGWR/serial_sgwr/alpha.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ALPHA")
    # Gaussian is only a default family marker in this Gaussian-only selector.
    # The complete ALPHA class, including all native bounds and search steps,
    # is executed unchanged. Avoid unrelated eager plotting/binary imports.
    class Gaussian:
        pass
    namespace = dict(np=np, cdist=cdist, Gaussian=Gaussian)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ALPHA"]


def fit_sgwr(data, output):
    idx, target = data.model_idx, data.test_idx
    x = features(data, idx)
    # All fitting covariates define similarity, as in the existing experiment.
    selector = official_alpha_class()(data.coords[idx], data.y[idx,None], x[idx],
                                       pd.DataFrame(x[idx]))
    bandwidth, alpha = selector.fit()
    pd.DataFrame(dict(bandwidth=[bandwidth], alpha=[alpha])).to_csv(
        Path(output,"native_selected.csv"), index=False)
    design = np.column_stack((np.ones(len(idx)), x[idx]))
    target_design = np.column_stack((np.ones(len(target)), x[target]))
    prediction = np.empty(len(target))
    for r, point in enumerate(target):
        distance = cdist(data.coords[point:point+1], data.coords[idx]).ravel()
        radius = np.partition(distance, int(bandwidth)-1)[int(bandwidth)-1]*1.0000001
        z = np.minimum(distance / radius, 1.0)
        geographical = (1-z*z)**2
        similarity = np.exp(-np.mean(np.abs(x[idx]-x[point]),axis=1)**2)
        # Preserve the published code's _build_wi row-sum normalisation.
        similarity /= similarity.sum()
        w = alpha*geographical+(1-alpha)*similarity
        weighted = design*np.sqrt(w[:,None])
        # No extra ridge, bounds, or coefficient clipping are added.
        gram = weighted.T@weighted
        beta = np.linalg.solve(gram, design.T@(w*data.y[idx]))
        prediction[r] = target_design[r]@beta
    return dict(test_prediction=prediction,
                selected=dict(bandwidth=float(bandwidth), alpha=float(alpha), criterion="AICc",
                              search="native ALPHA golden section and divide-and-conquer"),
                implementation="FastSGWR author ALPHA selector unchanged; held-out WLS adapter",
                protocol=dict(proposed_candidate_grid_used=False,
                              native_kernel="adaptive bisquare", added_ridge=False))
