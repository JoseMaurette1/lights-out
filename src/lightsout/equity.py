"""Equity analysis (Objective 3): do more socially vulnerable counties wait longer
for power after storms of comparable intensity?

1. Kaplan-Meier restoration curves by SVI quartile, within strata of storm
   intensity (peak % out tercile) - comparable storms compared to each other.
2. Cox model: restoration hazard ~ SVI quartile + storm/outage controls,
   stratified by state (a rough proxy for utility/regulatory differences).
   HR < 1 for Q4 vs Q1 means the most vulnerable counties are restored SLOWER.
3. Model-error disparity: are the severity/survival models less accurate for
   high-SVI counties?

This is observational: an association, not proof that utilities treat
communities differently (territory, terrain, tree cover and grid age all
correlate with SVI).
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import mean_absolute_error

from .models_survival import DURATION, EVENT, c_index, survival_frame

QUARTILES = ["Q1 (least)", "Q2", "Q3", "Q4 (most)"]


def with_intensity(df: pd.DataFrame) -> pd.DataFrame:
    s = survival_frame(df).dropna(subset=["svi_quartile"])
    s["intensity"] = pd.qcut(s["peak_pct"], 3, labels=["small", "medium", "large"])
    return s


def km_by_quartile(s: pd.DataFrame) -> dict:
    """{quartile: KaplanMeierFitter} for one subset."""
    out = {}
    for q in QUARTILES:
        d = s[s["svi_quartile"] == q]
        if len(d) >= 20:
            out[q] = KaplanMeierFitter(label=q).fit(d[DURATION], d[EVENT])
    return out


def logrank_by_stratum(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (inten, typ), d in s.groupby(["intensity", "storm_type"], observed=True):
        if d["svi_quartile"].nunique() < 2 or len(d) < 100:
            continue
        r = multivariate_logrank_test(d[DURATION], d["svi_quartile"].astype(str), d[EVENT])
        med = d.groupby("svi_quartile", observed=True)[DURATION].median()
        rows.append({"intensity": inten, "storm_type": typ, "n": len(d),
                     "median_h_Q1": med.get(QUARTILES[0], np.nan),
                     "median_h_Q4": med.get(QUARTILES[3], np.nan),
                     "chi2": r.test_statistic, "p_value": r.p_value})
    return pd.DataFrame(rows).sort_values(["intensity", "storm_type"])


def cox_equity(s: pd.DataFrame, svi_col: str = "svi_quartile") -> CoxPHFitter:
    controls = ["log_peak_pct", "max_wind_kts", "log_customers", "log_pop_density",
                "duration_h", "n_events"]
    X = s[controls + [DURATION, EVENT, "state"]].copy()
    X[controls] = X[controls].fillna(X[controls].median())
    X[controls] = (X[controls] - X[controls].mean()) / X[controls].std()
    if svi_col == "svi_quartile":
        X = X.join(pd.get_dummies(s[svi_col], prefix="svi", drop_first=True, dtype=float))
    else:
        X["svi_per_10pct"] = s[svi_col] * 10  # SVI is a 0-1 percentile
    X = X.join(pd.get_dummies(s["storm_type"], prefix="type", drop_first=True, dtype=float))
    X = X.loc[:, (X.nunique() > 1) | X.columns.isin(["state", EVENT])]
    return CoxPHFitter(penalizer=0.001).fit(X, DURATION, EVENT, strata=["state"])


def error_disparity(truth: pd.DataFrame, sev_pred: np.ndarray,
                    surv_truth: pd.DataFrame, surv_score: np.ndarray) -> pd.DataFrame:
    """Severity MAE and survival C-index per SVI quartile on the test set."""
    rows = []
    for q in QUARTILES:
        m = (truth["svi_quartile"] == q).to_numpy()
        ms = (surv_truth["svi_quartile"] == q).to_numpy()
        rows.append({
            "SVI quartile": q,
            "n (severity)": int(m.sum()),
            "severity MAE (log)": mean_absolute_error(truth.loc[m, "log_peak_pct"], sev_pred[m]) if m.any() else np.nan,
            "n (restoration)": int(ms.sum()),
            "restoration C-index": c_index(surv_truth[ms], surv_score[ms]) if ms.sum() > 10 else np.nan,
        })
    return pd.DataFrame(rows).set_index("SVI quartile").round(3)
