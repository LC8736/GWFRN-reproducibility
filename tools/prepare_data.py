"""Restore and validate source inputs; generate coverage directly from records.

Only the Python standard library is needed. Temperature chunks are byte ranges,
not independent CSVs: decompress in the manifest order and concatenate first.
No response-dependent filtering, imputation or model fitting takes place here.
"""
from pathlib import Path
import calendar
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/input"
OUT = ROOT / "generated/data"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def restore():
    manifest = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["other_files"]:
        if digest(INPUT / item["file"]) != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {item['file']}")
    target = INPUT / manifest["filename"]
    if target.exists():
        if digest(target) != manifest["sha256"]:
            raise ValueError("Existing fundat.csv differs from the frozen input.")
        return
    with tempfile.NamedTemporaryFile(dir=INPUT, delete=False) as f:
        temporary = Path(f.name)
        try:
            for item in manifest["parts"]:
                path = INPUT / item["file"]
                if digest(path) != item["sha256"]:
                    raise ValueError(f"Corrupt compressed chunk: {item['file']}")
                with gzip.open(path, "rb") as source:
                    shutil.copyfileobj(source, f)
        except BaseException:
            f.close(); temporary.unlink(missing_ok=True)
            raise
    if temporary.stat().st_size != manifest["bytes"] or digest(temporary) != manifest["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError("Reconstructed temperature matrix checksum failed.")
    os.replace(temporary, target)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    restore()
    rows = read(INPUT / "regdat_1.csv")
    locations = read(INPUT / "loc_info.csv")
    noaa = read(ROOT / "data/source/noaa_county_annual_precip.csv")
    source = {(r["CountyI"], r["Year"]): r for r in noaa}
    if len(source) != len(noaa):
        raise ValueError("Duplicate NOAA county-year record.")
    coverage = {r["CountyI"]: set() for r in locations}
    keys = set()
    years = {y: 0 for y in range(1999, 2019)}
    for r in rows:
        key = r["CountyI"], r["Year"]
        if key in keys or key[0] not in coverage:
            raise ValueError("Duplicate or unmatched regression key.")
        keys.add(key)
        precipitation = source[key]
        # Annual inches -> mean millimetres/day; retain leap-year convention.
        converted = float(precipitation["NOAA_annual_inches"]) * 25.4 / (366 if calendar.isleap(int(key[1])) else 365)
        if not math.isclose(converted, float(r["avgPRCP"]), rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError(f"Precipitation conversion mismatch: {key}")
        if int(key[1]) in years:
            if not math.isfinite(float(r["Yield"])) or float(r["Yield"]) <= 0:
                raise ValueError("Yield must be finite and positive before log transformation.")
            coverage[key[0]].add(int(key[1]))
            years[int(key[1])] += 1
    if len(rows) != 6340 or len(locations) != 403 or sum(years.values()) != 6152:
        raise ValueError("Unexpected source sample dimensions.")
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "county_coverage.csv", [dict(CountyI=k, n_years=len(v)) for k, v in coverage.items()])
    write(OUT / "annual_sample_sizes.csv", [dict(year=y, n=n) for y, n in years.items()])
    audit = dict(source_rows=len(rows), county_years_1999_2018=sum(years.values()),
                 distinct_counties=len(coverage), annual_n_min=min(years.values()),
                 annual_n_max=max(years.values()), years_per_county_min=min(map(len, coverage.values())),
                 years_per_county_max=max(map(len, coverage.values())),
                 precipitation_conversion="annual inches * 25.4 / days in calendar year",
                 temperature_sha256=digest(INPUT / "fundat.csv"))
    (OUT / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print("PASS input restoration, county/year keys, NOAA conversion and coverage.")


if __name__ == "__main__":
    main()
