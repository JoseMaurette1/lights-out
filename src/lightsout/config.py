"""Project-wide paths and analysis constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
# Big EAGLE-I yearly CSVs (~1 GB each) can live outside the project, e.g. on
# Colab's local disk, so only small processed outputs go to Google Drive.
EAGLEI_RAW = Path(os.environ.get("LIGHTSOUT_EAGLEI_RAW", RAW / "eaglei"))
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

YEARS = list(range(2014, 2023))
SVI_YEARS = [2014, 2016, 2018, 2020, 2022]

# Temporal split
TRAIN_YEARS = list(range(2014, 2020))
VAL_YEARS = [2020]
TEST_YEARS = [2021, 2022]

# Outage event detection (EAGLE-I, 15-minute snapshots).
# Bump EAGLEI_VERSION whenever detection logic changes so Colab rebuilds old years.
EAGLEI_VERSION = 2
STEP_MIN = 15
MIN_OUT_ABS = 100          # customers
MIN_OUT_FRAC = 0.005       # of modeled county customers
MERGE_GAP_H = 2.0          # merge outage events separated by less than this
RESTORE_FRAC = 0.10        # restored when customers_out <= 10% of peak (90% restored)...
SUSTAIN_H = 2.0            # ...and stays there for this long (filters scraper blips)
VALID_SNAPSHOT_FRAC = 0.5  # a 15-min snapshot is valid if it has >= 50% of the usual
                           # nationwide row count (rolling 24 h median); otherwise the
                           # scraper failed and missing rows are NOT treated as zero

# Storm-outage matching window around a NOAA episode in a county
MATCH_BEFORE_H = 3
MATCH_AFTER_H = 12

# Severity classes on peak % of customers out
SEV_BINS = [0.0, 0.01, 0.10]   # low < 1%, medium 1-10%, high >= 10%
SEV_LABELS = ["low", "medium", "high"]

# Case-study windows (UTC) saved at hourly resolution for plotting
CASE_STUDIES = {
    "texas_feb2021": ("2021-02-10", "2021-02-25"),
    "ida_aug2021": ("2021-08-28", "2021-09-15"),
}

for p in (RAW, INTERIM, PROCESSED, FIGURES, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
