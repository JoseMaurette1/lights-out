"""Restoration-time survival models (Objective 2).

Rows: (storm episode, county) units that had an outage.
Duration: restore_hours (outage start -> 90% of peak customers restored).
Event: restored (0 = censored: not restored within 14 days or data ran out).

Framing: predict restoration once the outage has peaked, so the peak size
(log_peak_pct) is a legitimate feature. SVI is deliberately left out of the
predictive models; it is studied separately in equity.py.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 42
DURATION, EVENT = "restore_hours", "restored"
SURV_NUMERIC = [c for c in NUMERIC_FEATURES if c not in ("county_hist_outage_rate",)] + ["log_peak_pct"]
SURV_CATEGORICAL = [c for c in CATEGORICAL_FEATURES if c != "state"]  # ~50 states is too many dummies for Cox


def survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df["had_outage"]].copy()
    s[DURATION] = s[DURATION].clip(lower=0.25)
    return s.reset_index(drop=True)


def design(df: pd.DataFrame, columns=None, stats=None):
    """Numeric + one-hot design matrix; impute/standardize with TRAIN stats."""
    X = pd.get_dummies(df[SURV_NUMERIC + SURV_CATEGORICAL], columns=SURV_CATEGORICAL,
                       drop_first=True, dtype=float)
    if stats is None:
        stats = (X[SURV_NUMERIC].median(), X[SURV_NUMERIC].std().replace(0, 1).fillna(1))
    med, sd = stats
    X[SURV_NUMERIC] = (X[SURV_NUMERIC].fillna(med) - med) / sd
    if columns is None:
        X = X.loc[:, X.std() > 0]  # constant columns break Cox
    else:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X, list(X.columns), stats


def fit_all(df: pd.DataFrame, rsf_max_train: int = 20000) -> dict:
    s = survival_frame(df)
    tr, va, te = (s[s["split"] == k].reset_index(drop=True) for k in ("train", "val", "test"))
    Xtr, cols, stats = design(tr)
    Xva, _, _ = design(va, cols, stats)
    Xte, _, _ = design(te, cols, stats)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(pd.concat([Xtr, tr[[DURATION, EVENT]]], axis=1), DURATION, EVENT)

    # RSF is O(n log n) per tree but memory-heavy; subsample training rows
    sub = tr.sample(min(rsf_max_train, len(tr)), random_state=SEED).index
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                               n_jobs=-1, random_state=SEED)
    rsf.fit(Xtr.loc[sub].to_numpy(), Surv.from_arrays(tr.loc[sub, EVENT].astype(bool),
                                                      tr.loc[sub, DURATION]))
    return {"cph": cph, "rsf": rsf, "X": {"train": Xtr, "val": Xva, "test": Xte},
            "data": {"train": tr, "val": va, "test": te}}


def risk_scores(fit: dict, split: str) -> dict:
    """Higher score = faster restoration (higher hazard of being restored)."""
    X = fit["X"][split]
    return {"Cox PH": fit["cph"].predict_partial_hazard(X).to_numpy(),
            "Random survival forest": fit["rsf"].predict(X.to_numpy())}


def c_index(d: pd.DataFrame, score: np.ndarray) -> float:
    # lifelines expects larger score = longer survival, so negate the hazard
    return concordance_index(d[DURATION], -score, d[EVENT])


def results_table(fit: dict, split: str = "test") -> pd.DataFrame:
    tr, d = fit["data"]["train"], fit["data"][split]
    scores = risk_scores(fit, split)
    y_tr = Surv.from_arrays(tr[EVENT].astype(bool), tr[DURATION])
    y_te = Surv.from_arrays(d[EVENT].astype(bool), d[DURATION])
    lo, hi = np.percentile(d.loc[d[EVENT] == 1, DURATION], [5, 80])
    hi = min(hi, tr[DURATION].max() * 0.99, d[DURATION].max() * 0.99)
    times = np.linspace(lo, hi, 30)
    X = fit["X"][split]
    surv = {
        "Cox PH": fit["cph"].predict_survival_function(X, times=times).T.to_numpy(),
        "Random survival forest": np.vstack([fn(times) for fn in
                                             fit["rsf"].predict_survival_function(X.to_numpy())]),
    }
    rows = [{"model": "Random (no skill)", "C-index": 0.5, "Integrated Brier": np.nan}]
    for name, sc in scores.items():
        rows.append({"model": name, "C-index": c_index(d, sc),
                     "Integrated Brier": integrated_brier_score(y_tr, y_te, surv[name], times)})
    return pd.DataFrame(rows).set_index("model").round(3)
