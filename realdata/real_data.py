"""Prepare annual county data and aligned functional inputs.

``fundat.csv`` has no identifiers. Its rows are paired positionally with
``regdat_1.csv`` before any filtering or sorting. Curves are observed on
days 1--365 and projected onto a common B-spline space.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from data import ANALYSIS_BASIS_DIM, T_GRID, bspline_coefficients


YEARS = tuple(range(1999, 2019))
METHODS = (
    "GWR", "SGWR", "FNN", "GNNWR", "SANNWR", "GWFRN", "SGWFRN"
)
N_DAYS = 365
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class AnnualData:
    year: int
    county_id: np.ndarray
    state: np.ndarray
    county: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    coords: np.ndarray
    x: np.ndarray
    z: np.ndarray
    y: np.ndarray
    bspline_coef: np.ndarray
    source_row: np.ndarray
    outer_fold: np.ndarray


@dataclass(frozen=True)
class ModelInput:
    """The fields consumed by the annual comparison estimators."""

    coords: np.ndarray
    x: np.ndarray
    bspline_coef: np.ndarray
    z: np.ndarray
    y: np.ndarray
    model_idx: np.ndarray
    test_idx: np.ndarray
    cv_folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def _finite(value: str, label: str, row_number: int) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Missing/non-numeric {label} in data row {row_number}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label} in data row {row_number}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_coordinates(locations: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Local equirectangular projection in kilometres for the Midwest region."""

    latitudes = np.array([item["latitude"] for item in locations.values()])
    longitudes = np.array([item["longitude"] for item in locations.values()])
    latitude_origin = np.deg2rad(float(latitudes.mean()))
    longitude_origin = float(longitudes.mean())
    latitude_origin_deg = float(latitudes.mean())
    projected = {}
    for county_id, item in locations.items():
        east = (
            EARTH_RADIUS_KM
            * np.cos(latitude_origin)
            * np.deg2rad(item["longitude"] - longitude_origin)
        )
        north = EARTH_RADIUS_KM * np.deg2rad(item["latitude"] - latitude_origin_deg)
        projected[county_id] = (float(east), float(north))
    return projected


def spatial_folds(coords: np.ndarray, identifiers: np.ndarray, n_folds: int) -> np.ndarray:
    """Balanced, deterministic contiguous spatial blocks by recursive bisection.

    Each terminal block has floor(n/K) or ceil(n/K) locations. Splits use the
    longest projected coordinate axis and are determined without the response.
    """

    n = len(coords)
    if not 2 <= n_folds <= n:
        raise ValueError("n_folds must lie between two and the sample size")
    result = np.full(n, -1, dtype=np.int64)

    def split(indices: np.ndarray, first_fold: int, count: int) -> None:
        if count == 1:
            result[indices] = first_fold
            return
        left_count = count // 2
        target_left = len(indices) * left_count // count
        if target_left < left_count or len(indices) - target_left < count - left_count:
            raise ValueError("Insufficient observations for balanced spatial blocks")
        span = np.ptp(coords[indices], axis=0)
        axis = int(np.argmax(span))
        # Lexical county ID breaks coordinate ties without depending on source order.
        order = np.lexsort((identifiers[indices], coords[indices, axis]))
        ordered = indices[order]
        split(ordered[:target_left], first_fold, left_count)
        split(ordered[target_left:], first_fold + left_count, count - left_count)

    split(np.arange(n, dtype=np.int64), 0, n_folds)
    sizes = np.bincount(result, minlength=n_folds)
    if sizes.max() - sizes.min() > 1:
        raise AssertionError(f"Unbalanced spatial folds: {sizes.tolist()}")
    return result


def prepare_data(data_dir: Path, cache_dir: Path, *, overwrite: bool = False) -> dict:
    """Write validated annual NPZ caches; never alter the supplied CSV files."""

    if len(T_GRID) != N_DAYS or T_GRID[0] != 1.0 or T_GRID[-1] != float(N_DAYS):
        raise ValueError("Functional observation grid must index days 1--365")
    data_dir = Path(data_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    sources = {name: data_dir / name for name in ("regdat_1.csv", "fundat.csv", "loc_info.csv")}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    signature = {name: _sha256(path) for name, path in sources.items()}
    manifest_path = cache_dir / "input_manifest.json"
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("sha256") != signature:
            raise ValueError("Source CSV changed; use --overwrite-cache only after reviewing the change")
        if (
            existing.get("time_domain_days") != [1, N_DAYS]
            or existing.get("functional_basis_dimension") != ANALYSIS_BASIS_DIM
        ):
            raise ValueError("Cached functional preprocessing differs from this run")
        if all((cache_dir / f"annual_{year}.npz").exists() for year in YEARS):
            return existing

    locations = {}
    with sources["loc_info.csv"].open(newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            county_id = row["CountyI"].strip()
            if county_id in locations:
                raise ValueError(f"Duplicate CountyI in loc_info.csv: {county_id}")
            latitude = _finite(row["latitude"], "latitude", row_number)
            raw_longitude = _finite(row["longitude"], "longitude", row_number)
            # Source stores the absolute value of west longitude.
            longitude = -abs(raw_longitude)
            locations[county_id] = {"latitude": latitude, "longitude": longitude}
    if len(locations) != 403:
        raise ValueError(f"Expected 403 unique location IDs, found {len(locations)}")
    projected = _project_coordinates(locations)

    by_year: dict[int, list[dict]] = {year: [] for year in YEARS}
    seen_pairs = set()
    row_count = 0
    with sources["regdat_1.csv"].open(newline="", encoding="utf-8-sig") as reg_stream, sources[
        "fundat.csv"
    ].open(newline="", encoding="utf-8-sig") as func_stream:
        reg_reader = csv.DictReader(reg_stream)
        func_reader = csv.reader(func_stream)
        header = next(func_reader)
        if len(header) != 2 * N_DAYS:
            raise ValueError(f"Expected 730 functional columns, found {len(header)}")
        for source_row, (reg_row, func_row) in enumerate(
            zip_longest(reg_reader, func_reader), start=1
        ):
            if reg_row is None or func_row is None:
                raise ValueError(f"Functional/regression row mismatch at data row {source_row}")
            row_count += 1
            year = int(reg_row["Year"])
            county_id = reg_row["CountyI"].strip()
            if county_id not in locations:
                raise ValueError(f"CountyI {county_id} has no coordinate record")
            if (year, county_id) in seen_pairs:
                raise ValueError(f"Duplicate county-year key: {(year, county_id)}")
            seen_pairs.add((year, county_id))
            if len(func_row) != 2 * N_DAYS:
                raise ValueError(f"Functional width {len(func_row)} in data row {source_row}")
            if year not in by_year:
                continue
            curve = np.array(
                [_finite(value, f"temperature V{column}", source_row)
                 for column, value in enumerate(func_row, start=1)],
                dtype=np.float64,
            ).reshape(2, N_DAYS)
            by_year[year].append(
                {
                    "county_id": county_id,
                    "state": reg_row["State"].strip(),
                    "county": reg_row["County"].strip(),
                    "latitude": locations[county_id]["latitude"],
                    "longitude": locations[county_id]["longitude"],
                    "coords": projected[county_id],
                    "x": curve,
                    "z": (
                        _finite(reg_row["avgPRCP"], "avgPRCP", source_row),
                        _finite(reg_row["Area"], "Area", source_row),
                    ),
                    "y": _finite(reg_row["Yield"], "Yield", source_row),
                    "source_row": source_row,
                }
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for year, records in by_year.items():
        if not records:
            raise ValueError(f"No observations in year {year}")
        records.sort(key=lambda row: int(row["county_id"]))
        x = np.stack([row["x"] for row in records])
        county_id = np.array([row["county_id"] for row in records])
        coords = np.array([row["coords"] for row in records], dtype=np.float64)
        outer_fold = spatial_folds(coords, county_id, 10)
        arrays = {
            "county_id": county_id,
            "state": np.array([row["state"] for row in records]),
            "county": np.array([row["county"] for row in records]),
            "latitude": np.array([row["latitude"] for row in records], dtype=np.float64),
            "longitude": np.array([row["longitude"] for row in records], dtype=np.float64),
            "coords": coords,
            "x": x,
            "z": np.array([row["z"] for row in records], dtype=np.float64),
            "y": np.array([row["y"] for row in records], dtype=np.float64),
            "bspline_coef": bspline_coefficients(x),
            "source_row": np.array([row["source_row"] for row in records], dtype=np.int64),
            "outer_fold": outer_fold,
        }
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, cache_dir / f"annual_{year}.npz")
        finally:
            temporary.unlink(missing_ok=True)
        counts[str(year)] = len(records)
    manifest = {
        "source_directory": "data/input",
        "sha256": signature,
        "regression_rows": row_count,
        "location_count": len(locations),
        "years": counts,
        "curve_days": N_DAYS,
        "time_domain_days": [1, N_DAYS],
        "functional_basis_dimension": ANALYSIS_BASIS_DIM,
        "functional_basis": f"{ANALYSIS_BASIS_DIM} orthonormalised cubic B-splines",
        "scalar_predictors": ["avgPRCP", "Area"],
        "outer_folds": 10,
        "spatial_split": "balanced recursive bisection of projected coordinates",
        "coordinate_projection": "local equirectangular kilometres, west longitudes negative",
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, delete=False, suffix=".tmp") as tmp:
        json.dump(manifest, tmp, indent=2, ensure_ascii=False)
        temporary = Path(tmp.name)
    os.replace(temporary, manifest_path)
    return manifest


def load_year(cache_dir: Path, year: int) -> AnnualData:
    with np.load(Path(cache_dir) / f"annual_{year}.npz", allow_pickle=False) as source:
        values = {name: source[name] for name in AnnualData.__dataclass_fields__ if name != "year"}
    return AnnualData(year=year, **values)


def make_fold_data(annual: AnnualData, fold: int, *, n_inner: int = 5) -> ModelInput:
    test_idx = np.flatnonzero(annual.outer_fold == fold).astype(np.int64)
    model_idx = np.flatnonzero(annual.outer_fold != fold).astype(np.int64)
    if len(test_idx) == 0 or len(model_idx) == 0:
        raise ValueError(f"Empty test or training set in year {annual.year}, fold {fold}")
    inner_labels = spatial_folds(
        annual.coords[model_idx], annual.county_id[model_idx], n_inner
    )
    cv_folds = tuple(
        (
            model_idx[inner_labels != inner],
            model_idx[inner_labels == inner],
        )
        for inner in range(n_inner)
    )
    return ModelInput(
        coords=annual.coords,
        x=annual.x,
        bspline_coef=annual.bspline_coef,
        z=annual.z,
        y=annual.y,
        model_idx=model_idx,
        test_idx=test_idx,
        cv_folds=cv_folds,
    )
