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

## How to run
1. **Build outage events in Colab** (avoids the ~8.5 GB EAGLE-I download on your laptop): `make colab-zip`, upload `LightsOut_code.zip` to My Drive, open `notebooks/00_colab_build_events.ipynb` in Colab, and Run all.
2. **Analysis**: run `notebooks/01_eda` … `04_equity` in Colab (they detect Drive automatically), or locally after copying `data/interim` and `data/processed` from Drive:
   ```
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS also: brew install libomp
   make models
   ```
Figures go to `figures/` and result tables to `reports/`.
