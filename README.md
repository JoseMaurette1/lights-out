# Lights Out

Predicting power-outage severity and restoration time after severe weather, and testing whether socially vulnerable counties wait longer. Jose Maurette & Daniel Bencomo.

## Pipeline
```
NOAA Storm Events ─┐
NWS zone→county ───┼─► noaa.py ─► storm_events_county ─┐
                   │                                    ├─► match.py ─► features.py ─► model_table.parquet
EAGLE-I (15-min) ──┴─► eaglei.py ─► outage_events_YYYY ─┘                    ▲
CDC SVI ─────────────────────────────────────────────────────── svi.py ─────┘
```
| Module | Role |
|---|---|
| `src/lightsout/config.py` | Paths, years, split, event and matching thresholds |
| `download.py` | Small datasets (~110 MB). EAGLE-I yearly files are fetched in Colab |
| `eaglei.py` | 15-min snapshots → outage events (start, peak, 90%-restored time, censoring) |
| `noaa.py` | Storm events → UTC, zone→county expansion, magnitudes |
| `match.py` | (storm episode, county) units + matched outage (peak = 0 if none) |
| `features.py` | Targets, features, train/val/test split |
| `models_severity.py`, `models_survival.py`, `equity.py` | Objectives 1–3 |
| `viz.py` | Shared figure style |

## Data in this repository
The processed dataset is included, so the analysis notebooks run without rebuilding anything. Raw downloads (~180 MB, plus ~8.5 GB of EAGLE-I) are not stored; `make download` and notebook 00 re-create them.

| File | Contents |
|---|---|
| `data/processed/model_table.parquet` | **The modeling table**: 226,389 (storm episode, county) units, 3,037 counties, 2014–2022, with targets, features, SVI and the train/val/test split |
| `data/interim/outage_events_YYYY.parquet` | Detected outage events per county and year (start, peak, 90%-restored time, censoring flag) |
| `data/interim/storm_events_county.parquet` | NOAA storm events expanded to counties, in UTC |
| `data/interim/units_matched.parquet` | Storm episodes × counties with their matched outage |
| `data/interim/daily_YYYY.parquet` | Daily maximum customers out per county |
| `data/interim/case_texas_feb2021.parquet`, `case_ida_aug2021.parquet` | 15-minute series for the two case studies |

Sources: [EAGLE-I](https://figshare.com/articles/dataset/The_Environment_for_Analysis_of_Geo-Located_Energy_Information_s_Recorded_Electricity_Outages_2014-2022/24237376) (ORNL), [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/), [CDC/ATSDR SVI](https://www.atsdr.cdc.gov/placeandhealth/svi/).

## Results (test years 2021–2022, models refit on 2014–2020)
| Task | Best model | Result (95% CI, bootstrap over storm episodes) |
|---|---|---|
| Severity class (low / medium / high) | LightGBM + CatBoost ensemble | accuracy 0.746 (0.739–0.751) vs 0.719 always-"low"; macro-F1 0.568 (0.559–0.576) |
| Severity, % of customers out | LightGBM + CatBoost ensemble | RMSE 0.811 on log(1 + %), MAE 2.64 percentage points |
| High-severity outages (≥10% out) | ensemble, cutoff fit on 2020 | AUC 0.852, AP 0.385; recall 0.443 at precision 0.393 |
| Restoration time | LightGBM + CatBoost on log hours | C-index 0.712 (0.706–0.717); integrated Brier 0.151; median error 3.1 h |
| Equity | Cox, state-stratified | most vs least vulnerable SVI quartile: restoration hazard ratio 0.915 (0.891–0.940) |

Extra data (`src/lightsout/extra_data.py`, all public): NOAA nClimGrid-Daily county weather, Iowa Environmental Mesonet ASOS station gusts, and EIA-861 utility ownership and reliability (previous year). On 2020 validation they raised high-severity average precision from 0.376 to 0.419; `reports/experiments_*.csv` has every comparison.
Hyperparameters were chosen by rolling-origin validation on 2019 and 2020 (`src/lightsout/tuning.py`, every trial in `reports/tuning_*.csv`); the final values are in `reports/hyperparameters_*.md`.
Figures are in `figures/` (01–05 exploratory analysis, 06–08 severity, 09–10 restoration, 11–13 equity) and result tables in `reports/` (`severity_*`, `survival_*`, `equity_*`).

## How to run
1. **Analysis only** (uses the included data): create the environment, then run `notebooks/01_eda` … `04_equity` locally or in Colab.
   ```
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS also: brew install libomp
   make models
   ```
   In Colab (no local compute): clone the repo, `pip install polars lifelines scikit-survival shap lightgbm optuna tabulate`, then run each notebook with `LIGHTSOUT_PROJECT=<clone path> jupyter nbconvert --to notebook --execute --inplace notebooks/0X_*.ipynb` (after `python -m lightsout.extra_data`, about 10 min; 02–04 then take about 2 h in total on a free runtime). `make tune` reruns the hyperparameter search (several hours).
2. **Rebuild the data from scratch**: `make download` for the small datasets, then `make colab-zip`, upload `LightsOut_code.zip` to My Drive, open `notebooks/00_colab_build_events.ipynb` in Colab and Run all (~20–40 min; avoids the ~8.5 GB EAGLE-I download on your laptop).

Figures go to `figures/` and result tables to `reports/`.
