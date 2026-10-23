"""Turn EAGLE-I 15-minute county snapshots into discrete outage events.

EAGLE-I only stores rows where customers_out > 0, so a county missing from a
15-minute snapshot means zero customers out -- but only if that snapshot is
*valid*. When the scraper fails, whole snapshots are empty or half-empty
nationwide; those timestamps are ignored instead of read as "everyone restored".
Single-snapshot spikes (e.g. 285 -> 48,324 -> 221) are removed with a 3-point
rolling median per county before events are detected.

Event definition (per county):
  * threshold  = max(MIN_OUT_ABS, MIN_OUT_FRAC * modeled customers)
  * an event is a run of snapshots at/above threshold; runs separated by
    less than MERGE_GAP_H of valid snapshots are merged
  * peak       = max customers_out in the event
  * t90        = first valid snapshot after the peak from which customers_out
                 stays <= 10% of peak for SUSTAIN_H (90% of peak restored)
  * censored   = no such time within 14 days or before the data runs out

Usage: python -m lightsout.eaglei [year ...]
"""
import sys

import numpy as np
import polars as pl

from . import config as C

STEP_S = C.STEP_MIN * 60
MAX_LOOKAHEAD_S = 14 * 24 * 3600


def load_mcc() -> pl.DataFrame:
    return (
        pl.read_csv(C.RAW / "eaglei" / "MCC.csv", encoding="utf8-lossy", infer_schema=False)
        .rename(lambda c: c.strip("﻿"))
        .select(pl.col("County_FIPS").cast(pl.Int32, strict=False).alias("fips"),
                pl.col("Customers").cast(pl.Float64, strict=False).alias("customers"))
        .drop_nulls()
    )


def _scan(year: int) -> pl.LazyFrame:
    return (
        pl.scan_csv(C.EAGLEI_RAW / f"eaglei_outages_{year}.csv",
                    schema_overrides={"fips_code": pl.Utf8, "customers_out": pl.Float64,
                                      "run_start_time": pl.Utf8})
        .select(
            pl.col("fips_code").cast(pl.Int32, strict=False).alias("fips"),
            pl.col("run_start_time").str.to_datetime(strict=False).alias("time"),
            pl.col("customers_out").alias("out"),
        )
        .drop_nulls()
    )


def load_series(year: int) -> pl.DataFrame:
    """One year of snapshots plus a 3-day tail from the next year (for restoration)."""
    lf = _scan(year)
    nxt = C.EAGLEI_RAW / f"eaglei_outages_{year + 1}.csv"
    if nxt.exists():
        from datetime import datetime
        tail = _scan(year + 1).filter(pl.col("time") < datetime(year + 1, 1, 4))
        lf = pl.concat([lf, tail])
    df = (
        lf.group_by(["fips", "time"]).agg(pl.col("out").max())
        .sort(["fips", "time"])
        .collect()
    )
    return despike(df)


def despike(df: pl.DataFrame) -> pl.DataFrame:
    """3-point median per county; a missing neighbouring snapshot counts as 0."""
    step = pl.duration(minutes=C.STEP_MIN)
    same = lambda k: (pl.col("fips").shift(k) == pl.col("fips")) & (
        (pl.col("time") - pl.col("time").shift(k)).abs() == step)
    prev = pl.when(same(1)).then(pl.col("out").shift(1)).otherwise(0.0)
    nxt = pl.when(same(-1)).then(pl.col("out").shift(-1)).otherwise(0.0)
    cur = pl.col("out")
    med = pl.max_horizontal(pl.min_horizontal(prev, cur),
                            pl.min_horizontal(pl.max_horizontal(prev, cur), nxt))
    return df.with_columns(med.alias("out")).filter(pl.col("out") > 0)


def valid_snapshots(df: pl.DataFrame) -> np.ndarray:
    """Epoch seconds of 15-min snapshots whose nationwide row count looks normal."""
    counts = df.group_by("time").len()
    grid = pl.datetime_range(counts["time"].min(), counts["time"].max(),
                             f"{C.STEP_MIN}m", eager=True, time_unit=counts["time"].dtype.time_unit
                             ).alias("time").to_frame()
    g = grid.join(counts, on="time", how="left").with_columns(pl.col("len").fill_null(0))
    med = g["len"].rolling_median(window_size=96, center=True, min_samples=1)
    ok = (g["len"] > 0) & (g["len"] >= C.VALID_SNAPSHOT_FRAC * med)
    return g.filter(ok)["time"].dt.epoch("s").to_numpy()


def detect_events(df: pl.DataFrame, mcc: pl.DataFrame, year: int) -> pl.DataFrame:
    df = df.join(mcc, on="fips", how="left").with_columns(
        pl.max_horizontal(pl.lit(float(C.MIN_OUT_ABS)),
                          pl.col("customers").fill_null(0) * C.MIN_OUT_FRAC).alias("thr"),
        pl.col("time").dt.epoch("s").alias("ts"),
    )
    V = valid_snapshots(df)
    merge_steps = int(C.MERGE_GAP_H * 3600 / STEP_S)

    # --- segment above-threshold snapshots into events (gaps counted in valid snapshots)
    above = df.filter(pl.col("out") >= pl.col("thr"))
    vidx = np.searchsorted(V, above["ts"].to_numpy())
    above = above.with_columns(pl.Series("vidx", vidx)).with_columns(
        ((pl.col("fips") != pl.col("fips").shift(1))
         | ((pl.col("vidx") - pl.col("vidx").shift(1)) > merge_steps))
        .fill_null(True).cum_sum().alias("eid")
    )
    ev = (
        above.group_by("eid", maintain_order=True)
        .agg(
            pl.col("fips").first(),
            pl.col("ts").min().alias("start_ts"),
            pl.col("out").max().alias("peak"),
            pl.col("ts").sort_by("out", descending=True).first().alias("peak_ts"),
            pl.col("customers").first(),
            pl.len().alias("n_above"),
        )
    )

    # --- restoration (t90): scan valid snapshots after the peak; a county
    # missing from a valid snapshot has 0 out. Restored once the value stays
    # <= 10% of peak for SUSTAIN_H.
    key = df["fips"].to_numpy().astype(np.int64) * 10**11 + df["ts"].to_numpy()
    out = df["out"].to_numpy()
    sustain = max(1, int(C.SUSTAIN_H * 3600 / STEP_S))
    horizon = int(MAX_LOOKAHEAD_S / STEP_S)
    ev_f = ev["fips"].to_numpy().astype(np.int64)
    ev_peak_ts = ev["peak_ts"].to_numpy()
    ev_peak = ev["peak"].to_numpy()
    k0 = np.searchsorted(V, ev_peak_ts, side="right")  # first valid snapshot after the peak

    t90 = np.empty(len(ev), dtype=np.int64)
    censored = np.ones(len(ev), dtype=bool)
    for i in range(len(ev)):
        thr90 = C.RESTORE_FRAC * ev_peak[i]
        found = False
        for width in (192, horizon):  # look 2 days ahead first, then the full 14 days
            times = V[k0[i]:k0[i] + width]
            if len(times) < sustain:
                break
            q = ev_f[i] * 10**11 + times
            pos = np.minimum(np.searchsorted(key, q), len(key) - 1)
            vals = np.where(key[pos] == q, out[pos], 0.0)
            low = (vals <= thr90).astype(np.int32)
            run = np.convolve(low, np.ones(sustain, dtype=np.int32), "valid")
            hit = np.flatnonzero(run == sustain)
            if hit.size:
                t90[i] = times[hit[0]]; censored[i] = False; found = True
                break
            if len(times) < width:
                break
        if not found:
            t90[i] = V[min(k0[i] + horizon, len(V)) - 1] if k0[i] < len(V) else ev_peak_ts[i]

    ev = ev.with_columns(
        pl.Series("t90_ts", t90),
        pl.Series("censored", censored),
    ).with_columns(
        pl.from_epoch("start_ts", "s").alias("start"),
        pl.from_epoch("peak_ts", "s").alias("peak_time"),
        pl.from_epoch("t90_ts", "s").alias("t90_time"),
        ((pl.col("t90_ts") - pl.col("start_ts")) / 3600).alias("restore_hours"),
        ((pl.col("t90_ts") - pl.col("peak_ts")) / 3600).alias("peak_to_t90_hours"),
        (pl.col("peak") / pl.col("customers")).alias("peak_pct"),
    )
    return (
        ev.filter(pl.col("start").dt.year() == year)
        .with_columns(pl.lit(year).alias("year"))
        .drop("eid", "start_ts", "peak_ts", "t90_ts")
    )


def county_daily(df: pl.DataFrame) -> pl.DataFrame:
    """Per county-day: max customers out and customer-hours out (for maps and EDA)."""
    return df.group_by(["fips", pl.col("time").dt.date().alias("date")]).agg(
        pl.col("out").max().alias("max_out"),
        (pl.col("out").sum() * C.STEP_MIN / 60).alias("customer_hours"),
        pl.len().alias("n_snapshots"),
    ).sort(["fips", "date"])


def case_study_hourly(df: pl.DataFrame, year: int) -> None:
    from datetime import datetime
    for name, (a, b) in C.CASE_STUDIES.items():
        if int(a[:4]) != year:
            continue
        lo, hi = datetime.fromisoformat(a), datetime.fromisoformat(b)
        (df.filter(pl.col("time").is_between(lo, hi))
         .group_by(["fips", pl.col("time").dt.truncate("1h").alias("hour")])
         .agg(pl.col("out").max()).sort(["fips", "hour"])
         .write_parquet(C.INTERIM / f"case_{name}.parquet"))


def done_marker(year: int):
    return C.INTERIM / f"outage_events_{year}.done"


def is_current(year: int) -> bool:
    m = done_marker(year)
    return m.exists() and m.read_text().strip() == str(C.EAGLEI_VERSION)


def run(year: int) -> None:
    out_path = C.INTERIM / f"outage_events_{year}.parquet"
    mcc = load_mcc()
    df = load_series(year)
    ev = detect_events(df, mcc, year)
    ev.write_parquet(out_path)
    this_year = df.filter(pl.col("time").dt.year() == year)
    county_daily(this_year).write_parquet(C.INTERIM / f"daily_{year}.parquet")
    case_study_hourly(this_year, year)
    done_marker(year).write_text(str(C.EAGLEI_VERSION))
    print(f"{year}: {df.height:,} snapshots -> {ev.height:,} events "
          f"({ev['censored'].mean():.1%} censored)")


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or C.YEARS
    for y in years:
        if (C.EAGLEI_RAW / f"eaglei_outages_{y}.csv").exists():
            run(y)
        else:
            print(f"{y}: raw file missing, skipped")
