"""Download raw data. Idempotent: existing files are skipped.

Usage:
  python -m lightsout.download                 # small files only (~110 MB)
  python -m lightsout.download --eaglei 2021   # one EAGLE-I year (~1 GB each)
"""
import json
import sys
import re
import shutil
import urllib.request
import zipfile

from .config import EAGLEI_RAW, RAW, SVI_YEARS, YEARS

FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/24237376"
NOAA_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NWS_ZONE_PAGE = "https://www.weather.gov/gis/ZoneCounty"
SVI_URL = "https://svi.cdc.gov/Documents/Data/{y}/csv/states_counties/SVI_{y}_US_county.csv"
COUNTY_SHP = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip"

UA = {"User-Agent": "Mozilla/5.0 (LightsOut student project)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def fetch(url: str, dest) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {dest.name}")
        return
    print(f"  get  {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def _figshare_files() -> dict:
    return {f["name"]: f["download_url"] for f in json.loads(_get(FIGSHARE_ARTICLE))["files"]}


def eaglei_meta() -> None:
    print("EAGLE-I metadata (customer counts, coverage)")
    files = _figshare_files()
    for name in ["MCC.csv", "DQI.csv", "coverage_history.csv"]:
        fetch(files[name], RAW / "eaglei" / name)


def eaglei_year(year: int) -> None:
    """One year of 15-minute outage snapshots (~1 GB) into EAGLEI_RAW."""
    name = f"eaglei_outages_{year}.csv"
    fetch(_figshare_files()[name], EAGLEI_RAW / name)


def noaa() -> None:
    print("NOAA Storm Events")
    listing = _get(NOAA_DIR).decode()
    for y in YEARS:
        names = sorted(set(re.findall(rf"StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz", listing)))
        fetch(NOAA_DIR + names[-1], RAW / "noaa" / f"details_{y}.csv.gz")


def zones() -> None:
    print("NWS zone-county correlation")
    page = _get(NWS_ZONE_PAGE).decode(errors="ignore")
    for href in sorted(set(re.findall(r'href="(/source/gis/Shapefiles/County/bp\w+\.dbx)"', page))):
        fetch("https://www.weather.gov" + href, RAW / "geo" / href.rsplit("/", 1)[1])


def svi() -> None:
    print("CDC SVI")
    for y in SVI_YEARS:
        fetch(SVI_URL.format(y=y), RAW / "svi" / f"SVI_{y}_US_county.csv")


def counties() -> None:
    print("Census county shapes")
    z = RAW / "geo" / "cb_2020_us_county_20m.zip"
    fetch(COUNTY_SHP, z)
    out = RAW / "geo" / "counties"
    if not out.exists():
        zipfile.ZipFile(z).extractall(out)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--eaglei":
        for y in (map(int, args[1:]) if args[1:] else YEARS):
            eaglei_year(y)
        return
    noaa()
    zones()
    svi()
    counties()
    eaglei_meta()


if __name__ == "__main__":
    main()
