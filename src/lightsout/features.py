"""Assemble the modeling table: one row per (storm episode, county).

Targets
  peak_pct       peak customers out / modeled customers (0 if no outage)
  severity       low (<1%) / medium (1-10%) / high (>=10%)
  restore_hours  outage start -> 90% restored (only rows with an outage)
  restored       1 if restoration observed, 0 if censored

Features are things known at (or forecastable before) storm time. NOAA
damage, injuries and deaths are kept for EDA but never used as features.

Usage: python -m lightsout.features
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C
from .eaglei import load_mcc
from .noaa import CATEGORY_PRIORITY
from .svi import svi_for_years

NUMERIC_FEATURES = [
    "max_wind_kts", "max_hail_in", "max_tor_scale", "tor_length_mi", "n_events",
    "duration_h", "episode_n_counties", "log_customers", "log_pop_density",
    "lat", "lon", "month_sin", "month_cos", "hour_sin", "hour_cos",
    "county_hist_outage_rate", "county_hist_log_peak",
    *[f"has_{c}" for c in CATEGORY_PRIORITY],
]
CATEGORICAL_FEATURES = ["storm_type", "season", "state"]
SVI_COLS = ["svi", "svi_socioeconomic", "svi_household", "svi_minority", "svi_housing_transport"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def county_centroids() -> pd.DataFrame:
    g = gpd.read_file(C.RAW / "geo" / "counties" / "cb_2020_us_county_20m.shp")
    pts = g.to_crs(5070).geometry.centroid.to_crs(4326)
    return pd.DataFrame({"fips": g["GEOID"].astype(int), "lat": pts.y, "lon": pts.x})


def split_of(year: int) -> str:
    if year in C.TRAIN_YEARS:
        return "train"
    if year in C.VAL_YEARS:
        return "val"
    return "test"


def build() -> pd.DataFrame:
    df = pd.read_parquet(C.INTERIM / "units_matched.parquet")
    df = df.drop(columns=["customers"], errors="ignore").merge(load_mcc().to_pandas(), on="fips", how="left")
    df = df[df["customers"] > 0]

    # --- targets
    df["peak_pct"] = (df["peak"] / df["customers"])
    df = df[df["peak_pct"] <= 1.0]   # >100% means a broken customer denominator
    df["log_peak_pct"] = np.log1p(100 * df["peak_pct"])  # log(1 + percent)
    df["severity"] = pd.cut(df["peak_pct"], bins=[-np.inf, *C.SEV_BINS[1:], np.inf],
                            right=False, labels=C.SEV_LABELS)
    df["restored"] = (~df["censored"].astype("boolean").fillna(True)).astype(int)

    # --- time / place features
    df["month"] = df["begin"].dt.month
    df["season"] = df["month"].map(SEASONS)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    hr = df["begin"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["log_customers"] = np.log(df["customers"])
    df = df.merge(county_centroids(), on="fips", how="left")
    df = df.merge(svi_for_years(sorted(df["year"].unique())), on=["fips", "year"], how="left")
    df["log_pop_density"] = np.log1p(df["population"] / df["area_sqmi"])
    for c in ["max_wind_kts", "max_hail_in", "max_tor_scale"]:
        df[c] = df[c].fillna(0)  # absent = not that hazard
    df["split"] = df["year"].map(split_of)

    # --- county history, computed from TRAINING years only (no leakage)
    tr = df[df["split"] == "train"]
    hist = tr.groupby("fips").agg(county_hist_outage_rate=("had_outage", "mean"),
                                  county_hist_log_peak=("log_peak_pct", "mean"))
    df = df.merge(hist, on="fips", how="left")
    for c in hist.columns:
        df[c] = df[c].fillna(hist[c].mean())  # counties unseen in training get the average

    df["svi_quartile"] = pd.qcut(df["svi"], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    return df


def main() -> None:
    df = build()
    df.to_parquet(C.PROCESSED / "model_table.parquet", index=False)
    print(f"model table: {len(df):,} rows, {df['fips'].nunique():,} counties")
    print(df.groupby("split").agg(n=("fips", "size"), outage_rate=("had_outage", "mean"),
                                  years=("year", lambda s: f"{s.min()}-{s.max()}")).to_string())
    print(df["severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
