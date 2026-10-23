"""CDC/ATSDR Social Vulnerability Index, county level.

Each storm year is mapped to the most recent SVI release at or before it
(2014-15 -> 2014, 2016-17 -> 2016, ...). SVI uses -999 for missing values.
"""
import pandas as pd

from . import config as C

THEMES = {
    "RPL_THEMES": "svi",
    "RPL_THEME1": "svi_socioeconomic",
    "RPL_THEME2": "svi_household",
    "RPL_THEME3": "svi_minority",
    "RPL_THEME4": "svi_housing_transport",
}


def load_release(year: int) -> pd.DataFrame:
    df = pd.read_csv(C.RAW / "svi" / f"SVI_{year}_US_county.csv", encoding="utf-8-sig",
                     usecols=["FIPS", "E_TOTPOP", "AREA_SQMI", *THEMES])
    df = df.rename(columns={"FIPS": "fips", "E_TOTPOP": "population",
                            "AREA_SQMI": "area_sqmi", **THEMES})
    df = df.mask(df == -999)
    df["fips"] = df["fips"].astype(int)
    df["svi_release"] = year
    return df


def svi_for_years(years) -> pd.DataFrame:
    """Long table (fips, year) -> SVI columns, using the nearest prior release."""
    releases = {y: load_release(y) for y in C.SVI_YEARS}
    out = []
    for y in years:
        rel = max(r for r in C.SVI_YEARS if r <= y)
        out.append(releases[rel].assign(year=y))
    return pd.concat(out, ignore_index=True)
