"""Build (storm episode, county) units and attach the matching outage event.

* Unit of analysis: NOAA EPISODE_ID x county FIPS. Multiple NOAA events in the
  same episode and county are aggregated.
* An outage event matches a unit if it starts within
  [episode begin - MATCH_BEFORE_H, episode end + MATCH_AFTER_H]. Each outage
  event is assigned to at most one unit (the episode whose begin is closest);
  each unit keeps its largest matched outage.
* Units with no matching outage are kept with peak = 0 (no outage above
  threshold) so severity models see the storms that did NOT knock out power.
* Units are only kept where EAGLE-I has data for that county in that year,
  and from the start of EAGLE-I coverage (2014-11-01).

Usage: python -m lightsout.match
"""
import numpy as np
import pandas as pd

from . import config as C
from .noaa import CATEGORY_PRIORITY

EAGLEI_START = pd.Timestamp("2014-11-01")


def load_outages() -> pd.DataFrame:
    paths = sorted(C.INTERIM.glob("outage_events_*.parquet"))
    ev = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ev["outage_id"] = np.arange(len(ev))
    return ev


def county_years_with_data() -> pd.DataFrame:
    rows = []
    for p in sorted(C.INTERIM.glob("daily_*.parquet")):
        h = pd.read_parquet(p, columns=["fips"])
        y = int(p.stem.split("_")[1])
        rows.append(pd.DataFrame({"fips": h["fips"].unique(), "year": y}))
    return pd.concat(rows, ignore_index=True)


def build_units(storms: pd.DataFrame) -> pd.DataFrame:
    g = storms.groupby(["episode_id", "fips"])
    units = g.agg(
        begin=("begin", "min"), end=("end", "max"),
        state=("state", "first"),
        n_events=("event_id", "nunique"),
        max_wind_kts=("wind_kts", "max"),
        max_hail_in=("hail_in", "max"),
        max_tor_scale=("tor_scale", "max"),
        tor_length_mi=("tor_length", "sum"),
        any_zone=("cz_type", lambda s: (s == "Z").any()),
        injuries=("injuries_direct", "sum"),
        deaths=("deaths_direct", "sum"),
        damage_usd=("damage_usd", "sum"),
        event_types=("event_type", lambda s: "|".join(sorted(set(s)))),
    ).reset_index()
    cats = (storms.assign(v=1)
            .pivot_table(index=["episode_id", "fips"], columns="category", values="v",
                         aggfunc="max", fill_value=0)
            .reindex(columns=CATEGORY_PRIORITY, fill_value=0)
            .add_prefix("has_").reset_index())
    units = units.merge(cats, on=["episode_id", "fips"])
    flags = units[[f"has_{c}" for c in CATEGORY_PRIORITY]].to_numpy()
    units["storm_type"] = np.array(CATEGORY_PRIORITY)[flags.argmax(axis=1)]
    units["duration_h"] = (units["end"] - units["begin"]).dt.total_seconds() / 3600
    # episode footprint: how many counties the whole episode touched
    units["episode_n_counties"] = units.groupby("episode_id")["fips"].transform("size")
    units["year"] = units["begin"].dt.year
    return units


def match(units: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    cand = units[["episode_id", "fips", "begin", "end"]].merge(
        outages[["outage_id", "fips", "start", "peak"]], on="fips")
    lo = cand["begin"] - pd.Timedelta(hours=C.MATCH_BEFORE_H)
    hi = cand["end"] + pd.Timedelta(hours=C.MATCH_AFTER_H)
    cand = cand[(cand["start"] >= lo) & (cand["start"] <= hi)].copy()
    # one episode per outage: the one whose begin is closest to the outage start
    cand["dist"] = (cand["start"] - cand["begin"]).abs()
    cand = cand.sort_values("dist").drop_duplicates("outage_id")
    n_matched = cand.groupby(["episode_id", "fips"]).size().rename("n_outages")
    best = cand.sort_values("peak", ascending=False).drop_duplicates(["episode_id", "fips"])
    best = best[["episode_id", "fips", "outage_id"]].merge(
        outages.drop(columns=["fips", "year"]), on="outage_id")
    out = units.merge(best, on=["episode_id", "fips"], how="left").merge(
        n_matched, on=["episode_id", "fips"], how="left")
    out["n_outages"] = out["n_outages"].fillna(0).astype(int)
    out["had_outage"] = out["outage_id"].notna()
    out["peak"] = out["peak"].fillna(0.0)
    return out


def main() -> None:
    storms = pd.read_parquet(C.INTERIM / "storm_events_county.parquet")
    storms = storms[storms["begin"] >= EAGLEI_START]
    units = build_units(storms)
    n0 = len(units)
    units = units.merge(county_years_with_data(), on=["fips", "year"])
    print(f"units: {n0:,} -> {len(units):,} with EAGLE-I data for the county-year")
    outages = load_outages()
    m = match(units, outages)
    assert not m.duplicated(["episode_id", "fips"]).any()
    m.to_parquet(C.INTERIM / "units_matched.parquet", index=False)
    print(f"matched outage rate: {m['had_outage'].mean():.1%}")
    print(m.groupby("storm_type")["had_outage"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
