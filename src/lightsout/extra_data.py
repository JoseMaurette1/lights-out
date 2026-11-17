"""Extra data sources added for the final (all public, no account needed).

  nClimGrid-Daily (NOAA NCEI)  county-average daily precipitation, max / min temperature
  IEM ASOS daily (Iowa State)  airport-station daily peak wind gust, max sustained wind,
                               precipitation and snowfall
  EIA-861 (US EIA)             which utilities serve each county, their ownership type and
                               reliability (SAIDI / SAIFI); always the PREVIOUS year's report,
                               so a storm's own outages never feed its features

`fetch_all()` downloads the raw files into data/raw/<source>/ (cached; re-runs skip files
already there). `build_extra_features(df)` returns one row per (episode_id, fips) with the
new columns, computed only from weather on the storm's own days (plus the 7 days before) and
from the prior year's utility report.

Usage (Colab): python -m lightsout.extra_data   -> data/processed/extra_features.parquet
"""
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

STATES = {  # postal abbreviation -> Census state FIPS
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "DC": 11, "FL": 12,
    "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22,
    "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31,
    "NV": 32, "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40,
    "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50,
    "VA": 51, "WA": 53, "WV": 54, "WI": 55, "WY": 56,
}
NCG_VARS = ["prcp", "tmax", "tmin"]
NCG_URL = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/averages/{y}/{v}-{y}{m:02d}-cty-scaled.csv"
IEM_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?network={st}_ASOS"
           "&year1={y}&month1=1&day1=1&year2={y}&month2=12&day2=31&format=csv")
IEM_STATIONS = "https://mesonet.agron.iastate.edu/geojson/network/{st}_ASOS.geojson"
EIA_URL = "https://www.eia.gov/electricity/data/eia861/archive/zip/f861{y}.zip"
STATION_RADIUS_KM = 60   # stations within this distance of a county centroid describe it
PRIOR_DAYS = 7           # antecedent precipitation window (wet soil topples trees)
HEADERS = {"User-Agent": "lightsout-class-project (python-requests)"}

EXTRA_FEATURES = [
    # nClimGrid, storm days
    "wx_prcp_max_mm", "wx_prcp_sum_mm", "wx_tmax_max_c", "wx_tmin_min_c", "wx_freeze_prcp_mm",
    "wx_prcp_prior7_mm",
    # ASOS, storm days, stations within STATION_RADIUS_KM
    "asos_gust_max_kt", "asos_wind_max_kt", "asos_snow_max_in", "asos_prcp_max_in", "asos_n_stations",
    # EIA-861, previous year
    "eia_n_utilities", "eia_share_coop", "eia_share_muni", "eia_share_iou",
    "eia_saidi_med", "eia_saifi_med", "eia_saidi_nomed", "eia_caidi_med",
]


def _get(url: str, dest: Path, tries: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return dest
            if r.status_code == 404:
                return dest  # leave missing; caller handles
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {url}")


def fetch_all(years=None) -> None:
    years = years or C.YEARS
    first = min(years)
    # weather: storm days can start the day before Jan 1 and need 7 prior days
    for y in sorted(set(years) | {first - 1}):
        for m in range(1, 13):
            if y == first - 1 and m < 12:
                continue
            for v in NCG_VARS:
                _get(NCG_URL.format(y=y, v=v, m=m), C.RAW / "nclimgrid" / f"{v}-{y}{m:02d}.csv")
    for st in STATES:
        _get(IEM_STATIONS.format(st=st), C.RAW / "iem" / f"stations_{st}.geojson")
        for y in years:
            _get(IEM_URL.format(st=st, y=y), C.RAW / "iem" / f"{st}_{y}.csv")
            time.sleep(0.5)  # be polite to the Iowa State server
    for y in range(first - 1, max(years)):
        _get(EIA_URL.format(y=y), C.RAW / "eia861" / f"f861{y}.zip")


# ---------------------------------------------------------------- nClimGrid
def load_nclimgrid() -> pd.DataFrame:
    """Long table: fips, date, prcp (mm), tmax, tmin (deg C)."""
    frames = []
    for v in NCG_VARS:
        for f in sorted((C.RAW / "nclimgrid").glob(f"{v}-*.csv")):
            if f.stat().st_size == 0:
                continue
            raw = pd.read_csv(f, header=None, dtype=str)
            st = raw[2].str.slice(0, 2)
            fips = st.map(STATES).astype("Int64") * 1000 + raw[1].str.slice(-3).astype(int)
            y, m = int(raw[3].iloc[0]), int(raw[4].iloc[0])
            vals = raw.iloc[:, 6:37].apply(pd.to_numeric, errors="coerce").to_numpy()
            ndays = pd.Period(f"{y}-{m:02d}").days_in_month
            vals = vals[:, :ndays]
            vals[vals <= -999] = np.nan
            dates = pd.date_range(f"{y}-{m:02d}-01", periods=ndays, freq="D")
            frames.append(pd.DataFrame({
                "fips": np.repeat(fips.to_numpy(), ndays), "date": np.tile(dates, len(raw)),
                "var": v, "value": vals.ravel()}))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["fips"])
    wide = long.pivot_table(index=["fips", "date"], columns="var", values="value").reset_index()
    wide["fips"] = wide["fips"].astype(int)
    return wide


# ---------------------------------------------------------------- IEM ASOS
def load_iem() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(station-day table, station locations)."""
    days, stations = [], []
    cols = ["station", "day", "precip_in", "snow_in", "max_wind_speed_kts", "max_wind_gust_kts"]
    for st in STATES:
        g = C.RAW / "iem" / f"stations_{st}.geojson"
        if g.exists() and g.stat().st_size > 0:
            js = json.loads(g.read_text())
            stations += [{"station": f["id"], "lon": f["geometry"]["coordinates"][0],
                          "lat": f["geometry"]["coordinates"][1]} for f in js["features"] if f.get("geometry")]
        for f in sorted((C.RAW / "iem").glob(f"{st}_*.csv")):
            if f.stat().st_size == 0 or not f.open().readline().startswith("station"):
                continue  # e.g. DC has no ASOS network of its own ("ERROR: Invalid network")
            d = pd.read_csv(f, usecols=cols, na_values=["None", "M", "T"])
            days.append(d)
    sd = pd.concat(days, ignore_index=True)
    sd["date"] = pd.to_datetime(sd["day"])
    sd = sd.drop(columns="day").drop_duplicates(["station", "date"])
    return sd, pd.DataFrame(stations).drop_duplicates("station")


def county_station_pairs(centroids: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """All (fips, station) pairs within STATION_RADIUS_KM (haversine)."""
    lat1, lon1 = np.radians(centroids["lat"].to_numpy())[:, None], np.radians(centroids["lon"].to_numpy())[:, None]
    lat2, lon2 = np.radians(stations["lat"].to_numpy())[None, :], np.radians(stations["lon"].to_numpy())[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    km = 6371 * 2 * np.arcsin(np.sqrt(a))
    i, j = np.nonzero(km <= STATION_RADIUS_KM)
    return pd.DataFrame({"fips": centroids["fips"].to_numpy()[i], "station": stations["station"].to_numpy()[j]})


# ---------------------------------------------------------------- EIA-861
def _norm_county(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False)
    s = s.str.replace(r"\b(county|parish|borough|census area|city and borough|municipality)\b", "", regex=True)
    s = s.str.replace(r"^saint ", "st ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _read_sheet(z: zipfile.ZipFile, key: str) -> pd.DataFrame:
    name = next(n for n in z.namelist() if key.lower() in n.lower() and n.lower().endswith((".xlsx", ".xls")))
    raw = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=0, header=None)
    hdr = next(i for i in range(6) if raw.iloc[i].astype(str).str.contains("Utility Number").any())
    out = raw.iloc[hdr + 1:].copy()
    cols, seen = [], {}
    for c in raw.iloc[hdr].astype(str):  # keep the first of duplicated names (IEEE standard block)
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}__{seen[c]}")
    out.columns = cols
    return out


def _reliability(z: zipfile.ZipFile) -> pd.DataFrame:
    """Utility reliability by position: the column names change across years (2021 uses a
    three-row header), but the layout does not. Each utility reports under the IEEE block or,
    for many cooperatives, only the "Other Standard" block; take IEEE, else Other."""
    name = next(n for n in z.namelist() if "reliab" in n.lower())
    raw = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=0, header=None)
    hdr = next(i for i in range(6) if raw.iloc[i].astype(str).str.contains("Utility Number").any())
    cols = raw.iloc[hdr].astype(str).tolist()
    body = raw.iloc[hdr + 1:]
    num = lambda j: pd.to_numeric(body.iloc[:, j], errors="coerce")
    saidi = [j for j, c in enumerate(cols) if "SAIDI" in c]
    cust = [j for j, c in enumerate(cols) if "Customers" in c]
    out = pd.DataFrame({"uid": pd.to_numeric(body.iloc[:, cols.index("Utility Number")], errors="coerce"),
                        "state": body.iloc[:, cols.index("State")], "own": body.iloc[:, cols.index("Ownership")]})
    blocks = [(saidi[0], saidi[1], cust[0])]
    if len(saidi) >= 5:
        blocks.append((saidi[3], saidi[4], cust[1] if len(cust) > 1 else cust[0]))
    for k, (med, nomed, cu) in enumerate(blocks):
        vals = {"saidi_med": num(med), "saifi_med": num(med + 1), "caidi_med": num(med + 2),
                "saidi_nomed": num(nomed), "customers": num(cu)}
        for c, v in vals.items():
            out[c] = v.to_numpy() if k == 0 else out[c].fillna(pd.Series(v.to_numpy(), index=out.index))
    return out


def load_eia(name_to_fips: pd.DataFrame) -> pd.DataFrame:
    """County-year utility features; row year y describes report year y (use with storm year y+1)."""
    rows = []
    for f in sorted((C.RAW / "eia861").glob("f861*.zip")):
        if f.stat().st_size == 0:
            continue
        year = int(f.stem[-4:])
        z = zipfile.ZipFile(f)
        terr = _read_sheet(z, "Service_Territory")[["Utility Number", "State", "County"]].dropna()
        rel = _reliability(z)
        rel = rel.dropna(subset=["uid"]).drop_duplicates(["uid", "state"])
        terr = terr.rename(columns={"Utility Number": "uid", "State": "state", "County": "county"})
        terr["uid"] = pd.to_numeric(terr["uid"], errors="coerce")
        terr["key"] = _norm_county(terr["county"])
        terr = terr.merge(name_to_fips, left_on=["state", "key"], right_on=["state", "key"], how="inner")
        m = terr.merge(rel, on=["uid", "state"], how="left")
        own = m["own"].astype(str).str.lower()
        m["coop"] = own.str.contains("coop").astype(float)
        m["muni"] = own.str.contains("municipal|political|state").astype(float)
        m["iou"] = own.str.contains("investor").astype(float)
        w = m["customers"].fillna(0)

        def wavg(g, col):
            ok = g[col].notna() & (g[col] >= 0)
            ww = w.loc[g.index][ok]
            return np.average(g.loc[ok, col], weights=ww) if ok.any() and ww.sum() > 0 else (
                g.loc[ok, col].mean() if ok.any() else np.nan)
        agg = m.groupby("fips").apply(lambda g: pd.Series({
            "eia_n_utilities": g["uid"].nunique(), "eia_share_coop": g["coop"].mean(),
            "eia_share_muni": g["muni"].mean(), "eia_share_iou": g["iou"].mean(),
            "eia_saidi_med": wavg(g, "saidi_med"), "eia_saifi_med": wavg(g, "saifi_med"),
            "eia_saidi_nomed": wavg(g, "saidi_nomed"), "eia_caidi_med": wavg(g, "caidi_med")}))
        agg["report_year"] = year
        rows.append(agg.reset_index())
        print(f"EIA {year}: {terr['fips'].nunique()} counties matched", flush=True)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- features
def _unit_days(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (unit, calendar day the storm touches)."""
    u = df[["episode_id", "fips", "begin", "end"]].copy()
    start, stop = u["begin"].dt.normalize(), u["end"].fillna(u["begin"]).dt.normalize()
    n = ((stop - start).dt.days.clip(lower=0, upper=13) + 1).to_numpy()
    rep = u.loc[u.index.repeat(n)].reset_index(drop=True)
    rep["date"] = start.repeat(n).to_numpy() + pd.to_timedelta(np.concatenate([np.arange(k) for k in n]), "D")
    return rep[["episode_id", "fips", "date"]]


def build_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    days = _unit_days(df)
    keys = ["episode_id", "fips"]

    # nClimGrid
    ncg = load_nclimgrid().sort_values(["fips", "date"])
    ncg["prcp_prior7"] = ncg.groupby("fips")["prcp"].transform(
        lambda s: s.shift(1).rolling(PRIOR_DAYS, min_periods=3).sum())
    ncg["freeze_prcp"] = ncg["prcp"].where((ncg["tmin"] <= 0) & (ncg["tmax"] <= 3), 0.0)
    w = days.merge(ncg, on=["fips", "date"], how="left")
    first_day = days.groupby(keys)["date"].min().rename("d0").reset_index().merge(
        ncg[["fips", "date", "prcp_prior7"]], left_on=["fips", "d0"], right_on=["fips", "date"], how="left")
    out = w.groupby(keys).agg(wx_prcp_max_mm=("prcp", "max"), wx_prcp_sum_mm=("prcp", "sum"),
                              wx_tmax_max_c=("tmax", "max"), wx_tmin_min_c=("tmin", "min"),
                              wx_freeze_prcp_mm=("freeze_prcp", "sum")).reset_index()
    out = out.merge(first_day[keys + ["prcp_prior7"]].rename(columns={"prcp_prior7": "wx_prcp_prior7_mm"}),
                    on=keys, how="left")
    print(f"nClimGrid: weather for {out['wx_prcp_max_mm'].notna().mean():.1%} of units", flush=True)

    # ASOS
    sd, st = load_iem()
    cent = df[["fips", "lat", "lon"]].drop_duplicates("fips").dropna()
    pairs = county_station_pairs(cent, st)
    a = days.merge(pairs, on="fips").merge(sd, on=["station", "date"], how="inner")
    asos = a.groupby(keys).agg(asos_gust_max_kt=("max_wind_gust_kts", "max"),
                               asos_wind_max_kt=("max_wind_speed_kts", "max"),
                               asos_snow_max_in=("snow_in", "max"), asos_prcp_max_in=("precip_in", "max"),
                               asos_n_stations=("station", "nunique")).reset_index()
    out = out.merge(asos, on=keys, how="left")
    out["asos_n_stations"] = out["asos_n_stations"].fillna(0)
    print(f"ASOS: gust for {out['asos_gust_max_kt'].notna().mean():.1%} of units "
          f"({len(st)} stations, {pairs['fips'].nunique()} counties within {STATION_RADIUS_KM} km)", flush=True)

    # EIA-861, previous year
    names = []
    for f in sorted((C.RAW / "nclimgrid").glob("prcp-*.csv"))[:1]:
        raw = pd.read_csv(f, header=None, dtype=str)
        st_abbr = raw[2].str.slice(0, 2)
        names = pd.DataFrame({"state": st_abbr, "key": _norm_county(raw[2].str.slice(4)),
                              "fips": st_abbr.map(STATES).astype("Int64") * 1000 + raw[1].str.slice(-3).astype(int)})
    names = names.dropna().drop_duplicates(["state", "key"])
    names["fips"] = names["fips"].astype(int)
    eia = load_eia(names)
    eia["year"] = eia["report_year"] + 1
    yr = df[keys + ["year"]]
    out = yr.merge(out, on=keys, how="left").merge(eia.drop(columns="report_year"), on=["fips", "year"], how="left")
    print(f"EIA: utility features for {out['eia_n_utilities'].notna().mean():.1%} of units", flush=True)
    return out.drop(columns="year")


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "model_table.parquet")
    fetch_all()
    ex = build_extra_features(df)
    ex.to_parquet(C.PROCESSED / "extra_features.parquet", index=False)
    print(ex[EXTRA_FEATURES].describe().T.round(2).to_string())
    # add the columns to the modeling table (replacing any earlier version of them)
    df = df.drop(columns=[c for c in EXTRA_FEATURES if c in df.columns]).merge(ex, on=["episode_id", "fips"], how="left")
    df.to_parquet(C.PROCESSED / "model_table.parquet", index=False)
    print(f"model table: {len(df):,} rows, {len(df.columns)} columns", flush=True)


if __name__ == "__main__":
    main()
