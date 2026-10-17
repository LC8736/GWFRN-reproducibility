"""Shared numerical county-polygon rendering for manuscript Figures 2 and 3."""
from pathlib import Path
import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/figures"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def geography(aggregate):
    loc = pd.read_csv(ROOT / "data/input/loc_info.csv", dtype={"CountyI": str, "State.ANSI": str, "County.ANSI": str})
    loc["GEOID"] = loc["State.ANSI"].str.zfill(2) + loc["County.ANSI"].str.zfill(3)
    keyed = aggregate.merge(loc[["CountyI", "GEOID"]], on="CountyI", validate="one_to_one")
    if len(keyed) != 403:
        raise AssertionError("Expected all 403 county IDs.")
    polygons = []
    filename = ROOT / "data/geography/cb_2018_us_county_5m.shp"
    for item in shapefile.Reader(str(filename)).shapeRecords():
        state = str(item.record.STATEFP).zfill(2)
        if state in {"17", "18", "19", "20", "29"}:
            polygons.append(dict(GEOID=str(item.record.GEOID).zfill(5), STATEFP=state,
                                 geometry=shape(item.shape.__geo_interface__)))
    counties = gpd.GeoDataFrame(polygons, geometry="geometry")
    missing = set(keyed.GEOID) - set(counties.GEOID)
    if missing:
        raise AssertionError(f"County polygons missing: {sorted(missing)}")
    return counties, counties.dissolve(by="STATEFP"), counties.merge(keyed, on="GEOID", how="left", validate="one_to_one")


def plot_base(ax, counties, states, mapped, column, cmap, norm):
    valid = mapped[column].notna()
    if int(valid.sum()) != 403:
        raise AssertionError("A mapped county has no source value.")
    counties.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#dedede", linewidth=0.13)
    mapped.loc[valid].plot(ax=ax, column=column, cmap=cmap, norm=norm,
                           edgecolor="#f2f2f2", linewidth=0.11)
    states.boundary.plot(ax=ax, color="#4d4d4d", linewidth=0.55, zorder=3)
    ax.set_aspect(1 / np.cos(np.deg2rad(39.0)))
    ax.set_axis_off()


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    # No modeler/author metadata or private filesystem paths are embedded.
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white", metadata={"Author": "", "Title": stem})
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", metadata={"Creator": "Matplotlib"})
    fig.savefig(OUT / f"{stem}.png", dpi=350, facecolor="white", metadata={"Author": ""})
    plt.close(fig)
