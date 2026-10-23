# Lights Out — Final Report (OUTLINE)

> Skeleton to fill after notebooks 01–04 run on the full data. Figure and table files are referenced by name.

1. **Introduction**: problem, the three objectives, why it matters (crew staging, equity).
2. **Data**: EAGLE-I, NOAA, SVI; coverage caveats (from the midterm).
3. **Methods**
   - Event detection and matching (from the midterm, with final parameter values from `src/lightsout/config.py`).
   - Severity models and the temporal split; metrics.
   - Survival models: Cox PH (with PH check) and RSF; C-index, integrated Brier.
   - Equity design: intensity strata, state-stratified Cox, error disparity.
4. **Results**
   - Severity: `reports/severity_results_test.md`, `figures/06_*`, SHAP `07_*`, `08_*`.
   - Restoration: `reports/survival_results_test.md`, `figures/09_*`, `10_*`.
   - Equity: `figures/11_*`, `12_*`, `13_*`, `reports/equity_cox.md`, `reports/equity_logrank.csv`, `reports/equity_model_error.csv`.
5. **Discussion**: what drives severity; how predictable restoration is; size and direction of the SVI effect and the confounders.
6. **Limitations and future work**
7. **Contributions**: Jose (pipeline, survival); Daniel (severity, SHAP, equity); both (EDA, report, presentation).
