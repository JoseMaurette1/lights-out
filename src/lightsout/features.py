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
from .extra_data import EXTRA_FEATURES
from .noaa import CATEGORY_PRIORITY
from .svi import svi_for_years

NUMERIC_FEATURES = [
    "max_wind_kts", "max_hail_in", "max_tor_scale", "tor_length_mi", "n_events",
    "duration_h", "episode_n_counties", "log_customers", "log_pop_density",
    "lat", "lon", "month_sin", "month_cos", "hour_sin", "hour_cos",
    "county_hist_outage_rate", "county_hist_log_peak",
    "county_type_hist_outage_rate", "county_type_hist_log_peak",
    "state_hist_outage_rate", "state_hist_log_peak",
    "county_hist_log_restore", "county_type_hist_log_restore", "state_hist_log_restore",
    "prior_30d_units", "prior_30d_outages", "prior_365d_units", "prior_365d_outages",
    "prior_365d_outage_rate", "prior_365d_log_peak", "any_zone", "log_area",
    *[f"has_{c}" for c in CATEGORY_PRIORITY],
    *EXTRA_FEATURES,  # weather, station wind and utility data (extra_data.py)
]
HIST_SMOOTHING = 10  # pseudo-observations at the overall mean when encoding history
PRIOR_LAG = "48h"    # prior_* features skip storms that began < 48 h earlier (outcome not yet known)
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


def _smoothed_means(src: pd.DataFrame, keys: list, col: str) -> tuple[pd.Series, float]:
    g = src.groupby(keys)[col].agg(["size", "mean"])
    m0 = src[col].mean()
    return (g["size"] * g["mean"] + HIST_SMOOTHING * m0) / (g["size"] + HIST_SMOOTHING), m0


def _encode_history(df: pd.DataFrame, keys: list, prefix: str, train_years: list) -> pd.DataFrame:
    """Smoothed per-group history from the TRAINING years only:
      {prefix}_hist_outage_rate   share of storms that caused an outage
      {prefix}_hist_log_peak      mean log(1 + peak % out)
      {prefix}_hist_log_restore   mean log restoration hours (outages only)

    Training rows get leave-one-year-out values so a row's own outcome never
    feeds its feature; every other row uses all training years.
    """
    is_tr = df["year"].isin(train_years)
    tr = df[is_tr]
    out = {k: pd.Series(np.nan, index=df.index) for k in ("outage_rate", "log_peak", "log_restore")}

    def fill(src, idx):
        if len(idx) == 0:
            return
        keyed = df.loc[idx, keys]
        for name, col, rows in (("outage_rate", "had_outage", src),
                                ("log_peak", "log_peak_pct", src),
                                ("log_restore", "_log_restore", src[src["had_outage"]])):
            enc, m0 = _smoothed_means(rows, keys, col)
            m = keyed.merge(enc.rename("v"), left_on=keys, right_index=True, how="left")["v"]
            out[name][idx] = m.fillna(m0).to_numpy()  # groups unseen in training get the average

    for y in train_years:
        fill(tr[tr["year"] != y], df.index[is_tr & (df["year"] == y)])
    fill(tr, df.index[~is_tr])
    for k, v in out.items():
        df[f"{prefix}_hist_{k}"] = v
    return df


def _lagged_rolling(g, col: str, window: str, lag: str, how: str) -> np.ndarray:
    """Rolling count/sum over [begin - window, begin - lag): earlier storms whose
    outcome is already known when this storm begins."""
    full = getattr(g[col].rolling(window, closed="left"), how)().to_numpy()
    recent = getattr(g[col].rolling(lag, closed="left"), how)().to_numpy()
    return np.nan_to_num(full) - np.nan_to_num(recent)


def with_history(df: pd.DataFrame, train_years: list | None = None) -> pd.DataFrame:
    """(Re)compute the target-encoded *_hist_* features using `train_years` as the
    known past. Used to build the table (2014-2019), for the final refit on
    2014-2020, and for each fold of the rolling-origin tuning."""
    train_years = C.TRAIN_YEARS if train_years is None else train_years
    df = df.copy()
    df["_log_restore"] = np.log(df["restore_hours"].clip(lower=0.25))
    for keys, prefix in ((["fips"], "county"), (["fips", "storm_type"], "county_type"),
                         (["state"], "state")):
        df = _encode_history(df, keys, prefix, train_years)
    return df.drop(columns="_log_restore")


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past outage behaviour of the county / county x storm type / state.

    *_hist_*   target encodings from training years (see with_history)
    prior_*    storms and outages in the same county that began at least
               PRIOR_LAG before this storm (so their outcome is already known);
               only uses the past, so it is valid in every split
    """
    df = df.sort_values(["fips", "begin"]).reset_index(drop=True)
    df = with_history(df)

    df["_peak"] = df["log_peak_pct"].where(df["had_outage"])
    g = df.set_index("begin").groupby("fips", sort=False)
    for w in ("30d", "365d"):
        df[f"prior_{w}_units"] = _lagged_rolling(g, "had_outage", w.upper(), PRIOR_LAG, "count")
        df[f"prior_{w}_outages"] = _lagged_rolling(g, "had_outage", w.upper(), PRIOR_LAG, "sum")
    df["prior_365d_outage_rate"] = df["prior_365d_outages"] / df["prior_365d_units"].replace(0, np.nan)
    n_pk = _lagged_rolling(g, "_peak", "365D", PRIOR_LAG, "count")
    df["prior_365d_log_peak"] = _lagged_rolling(g, "_peak", "365D", PRIOR_LAG, "sum") / np.where(n_pk > 0, n_pk, np.nan)
    return df.drop(columns="_peak")


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

    df["any_zone"] = df["any_zone"].astype(float)
    df["log_area"] = np.log(df["area_sqmi"])
    df = add_history_features(df)

    df["svi_quartile"] = pd.qcut(df["svi"], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    extra = C.PROCESSED / "extra_features.parquet"
    if extra.exists():
        df = df.merge(pd.read_parquet(extra), on=["episode_id", "fips"], how="left")
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
