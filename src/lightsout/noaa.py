"""Load NOAA Storm Events, convert to UTC, and expand forecast zones to counties.

Output: data/interim/storm_events_county.parquet, one row per (NOAA event, county).

Usage: python -m lightsout.noaa
"""
import numpy as np
import pandas as pd

from . import config as C

# Event types that plausibly cause power outages, grouped into categories.
# The first category in CATEGORY_PRIORITY present in an episode is its primary type.
EVENT_CATEGORY = {
    "Hurricane": "tropical", "Hurricane (Typhoon)": "tropical", "Tropical Storm": "tropical",
    "Tropical Depression": "tropical", "Storm Surge/Tide": "tropical",
    "Ice Storm": "ice",
    "Tornado": "tornado",
    "Winter Storm": "winter", "Blizzard": "winter", "Heavy Snow": "winter",
    "Winter Weather": "winter", "Lake-Effect Snow": "winter",
    "Cold/Wind Chill": "winter", "Extreme Cold/Wind Chill": "winter", "Frost/Freeze": "winter",
    "Thunderstorm Wind": "wind", "High Wind": "wind", "Strong Wind": "wind",
    "Hail": "hail",
    "Flash Flood": "flood", "Flood": "flood", "Heavy Rain": "flood", "Coastal Flood": "flood",
    "Lightning": "lightning",
    "Excessive Heat": "heat", "Heat": "heat",
    "Wildfire": "wildfire",
}
CATEGORY_PRIORITY = ["tropical", "ice", "tornado", "winter", "wind", "hail",
                     "flood", "lightning", "heat", "wildfire"]
MAX_EVENT_DAYS = 7  # drop multi-week records (e.g. long-running floods) that can't be matched sensibly

DAMAGE_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "H": 1e2}


def parse_damage(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip().str.upper()
    num = pd.to_numeric(s.str.extract(r"^([\d.]+)")[0], errors="coerce")
    mult = s.str.extract(r"([KMBH])$")[0].map(DAMAGE_MULT).fillna(1.0)
    return (num * mult).fillna(0.0)


def load_zone_county() -> pd.DataFrame:
    cols = ["STATE", "ZONE", "CWA", "NAME", "STATE_ZONE", "COUNTY", "FIPS",
            "TIME_ZONE", "FE_AREA", "LAT", "LON"]
    frames = [pd.read_csv(p, sep="|", header=None, names=cols, dtype=str, encoding="latin-1")
              for p in sorted((C.RAW / "geo").glob("bp*.dbx"))]
    z = pd.concat(frames)
    z["state_fips"] = z["FIPS"].str[:2].astype(int)
    z["zone"] = z["ZONE"].astype(int)
    z["fips"] = z["FIPS"].astype(int)
    return z[["state_fips", "zone", "fips"]].drop_duplicates()


def load_details(year: int) -> pd.DataFrame:
    usecols = ["EPISODE_ID", "EVENT_ID", "STATE", "STATE_FIPS", "YEAR", "EVENT_TYPE",
               "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "BEGIN_DATE_TIME", "END_DATE_TIME",
               "CZ_TIMEZONE", "INJURIES_DIRECT", "DEATHS_DIRECT", "DAMAGE_PROPERTY",
               "DAMAGE_CROPS", "MAGNITUDE", "MAGNITUDE_TYPE", "TOR_F_SCALE",
               "TOR_LENGTH", "TOR_WIDTH", "BEGIN_LAT", "BEGIN_LON"]
    df = pd.read_csv(C.RAW / "noaa" / f"details_{year}.csv.gz", usecols=usecols,
                     low_memory=False)
    return df


def to_utc(local: pd.Series, tz: pd.Series) -> pd.Series:
    t = pd.to_datetime(local, format="%d-%b-%y %H:%M:%S")
    offset_h = pd.to_numeric(tz.astype(str).str.extract(r"(-?\d+)$")[0], errors="coerce").fillna(0)
    return t - pd.to_timedelta(offset_h, unit="h")


def build() -> pd.DataFrame:
    df = pd.concat([load_details(y) for y in C.YEARS], ignore_index=True)
    df = df[df["EVENT_TYPE"].isin(EVENT_CATEGORY)].copy()
    df["category"] = df["EVENT_TYPE"].map(EVENT_CATEGORY)
    df["begin"] = to_utc(df["BEGIN_DATE_TIME"], df["CZ_TIMEZONE"])
    df["end"] = to_utc(df["END_DATE_TIME"], df["CZ_TIMEZONE"])
    df = df[(df["end"] >= df["begin"])
            & ((df["end"] - df["begin"]) <= pd.Timedelta(days=MAX_EVENT_DAYS))]

    df["damage_usd"] = parse_damage(df["DAMAGE_PROPERTY"]) + parse_damage(df["DAMAGE_CROPS"])
    mag = pd.to_numeric(df["MAGNITUDE"], errors="coerce")
    is_wind = df["MAGNITUDE_TYPE"].isin(["EG", "MG", "ES", "MS"])
    df["wind_kts"] = np.where(is_wind, mag, np.nan)
    df["hail_in"] = np.where(df["EVENT_TYPE"] == "Hail", mag, np.nan)
    df["tor_scale"] = pd.to_numeric(
        df["TOR_F_SCALE"].astype(str).str.extract(r"(\d)$")[0], errors="coerce")

    # county events map directly; zone events expand to every county in the zone
    county = df[df["CZ_TYPE"] == "C"].copy()
    county["fips"] = county["STATE_FIPS"] * 1000 + county["CZ_FIPS"]
    zc = load_zone_county()
    zone = df[df["CZ_TYPE"] == "Z"].merge(
        zc, left_on=["STATE_FIPS", "CZ_FIPS"], right_on=["state_fips", "zone"], how="inner"
    ).drop(columns=["state_fips", "zone"])
    out = pd.concat([county, zone], ignore_index=True)
    out = out[out["STATE_FIPS"] < 60]  # 50 states + DC (drop territories and marine)

    keep = ["EPISODE_ID", "EVENT_ID", "fips", "STATE", "YEAR", "EVENT_TYPE", "category",
            "CZ_TYPE", "begin", "end", "wind_kts", "hail_in", "tor_scale", "TOR_LENGTH",
            "TOR_WIDTH", "INJURIES_DIRECT", "DEATHS_DIRECT", "damage_usd",
            "BEGIN_LAT", "BEGIN_LON"]
    out = out[keep].rename(columns=str.lower)
    out["fips"] = out["fips"].astype(int)
    return out


def main() -> None:
    ev = build()
    path = C.INTERIM / "storm_events_county.parquet"
    ev.to_parquet(path, index=False)
    print(f"{len(ev):,} (event, county) rows; {ev['episode_id'].nunique():,} episodes")
    print(ev.groupby("category").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
