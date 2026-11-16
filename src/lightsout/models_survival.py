"""Restoration-time survival models (Objective 2).

Rows: (storm episode, county) units that had an outage.
Duration: restore_hours (outage start -> 90% of peak customers restored).
Event: restored (0 = censored: not restored within 14 days or data ran out).

Framing: predict restoration once the outage has peaked, so the peak size
(log_peak_pct) is a legitimate feature. SVI is deliberately left out of the
predictive models; it is studied separately in equity.py.

Models
  Cox PH                      penalized linear proportional-hazards model
  Random survival forest      non-parametric, handles censoring natively
  LightGBM (log hours)        gradient boosting on log restoration hours. Only
                              ~0.1% of rows are censored, so treating censored
                              durations as observed changes almost nothing. Its
                              survival curve is S(t|x) = P(resid > log t - f(x)),
                              with residuals taken from the 2020 validation year
                              (an accelerated-failure-time model with an
                              empirical error distribution).

Same protocol as severity: stage 1 fits on 2014-2019 and is scored on 2020;
stage 2 refits on 2014-2020 (history features recomputed) and is scored on
2021-2022.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, with_history

SEED = 42
DURATION, EVENT = "restore_hours", "restored"
SURV_NUMERIC = [c for c in NUMERIC_FEATURES if c not in ("county_hist_outage_rate",)] + ["log_peak_pct"]
SURV_CATEGORICAL = [c for c in CATEGORICAL_FEATURES if c != "state"]  # ~50 states is too many dummies for Cox
FINAL_TRAIN_YEARS = C.TRAIN_YEARS + C.VAL_YEARS

# Chosen by tuning.py (rolling-origin validation on 2019 and 2020); see reports/tuning_best.csv
COX_PENALIZER, COX_L1 = 0.01, 0.0
RSF_PARAMS = dict(min_samples_leaf=7, max_features=0.33, max_depth=None, max_samples=0.43)
RSF_TREES = 100
LGB_SURV = dict(learning_rate=0.0103, num_leaves=34, min_child_samples=11, subsample=0.643,
                colsample_bytree=0.591, reg_lambda=0.165, min_split_gain=0.472)
MODELS = ["Cox PH", "Random survival forest", "LightGBM (log hours)"]


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


def _fit_models(tr: pd.DataFrame, Xtr: pd.DataFrame, va=None, Xva=None, lgb_iter=None) -> dict:
    cph = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=COX_L1)
    cph.fit(pd.concat([Xtr, tr[[DURATION, EVENT]]], axis=1), DURATION, EVENT)
    rsf = RandomSurvivalForest(n_estimators=RSF_TREES, **RSF_PARAMS, n_jobs=-1, random_state=SEED)
    rsf.fit(Xtr.to_numpy(), Surv.from_arrays(tr[EVENT].astype(bool), tr[DURATION]))
    kw = dict(subsample_freq=1, random_state=SEED, verbose=-1, **LGB_SURV)
    if va is not None:
        gbm = lgb.LGBMRegressor(n_estimators=5000, **kw).fit(
            Xtr, np.log(tr[DURATION]), eval_set=[(Xva, np.log(va[DURATION]))],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        gbm = lgb.LGBMRegressor(n_estimators=lgb_iter, **kw).fit(Xtr, np.log(tr[DURATION]))
    return {"cph": cph, "rsf": rsf, "gbm": gbm}


def fit_all(df: pd.DataFrame, refit: bool = True) -> dict:
    s = survival_frame(df)
    tr, va, te = (s[s["split"] == k].reset_index(drop=True) for k in ("train", "val", "test"))
    Xtr, cols, stats = design(tr)
    Xva, _, _ = design(va, cols, stats)
    m1 = _fit_models(tr, Xtr, va, Xva)
    resid = np.log(va[DURATION].to_numpy()) - m1["gbm"].predict(Xva)  # out-of-sample residuals

    out = {"stage1": m1, "resid": np.sort(resid), "X": {"val": Xva}, "data": {"val": va}}
    if refit:
        s2 = survival_frame(with_history(df, FINAL_TRAIN_YEARS))
        tr2 = s2[s2["year"].isin(FINAL_TRAIN_YEARS)].reset_index(drop=True)
        te = s2[s2["split"] == "test"].reset_index(drop=True)
        Xtr2, cols2, stats2 = design(tr2)
        lgb_iter = int(round(m1["gbm"].best_iteration_ * len(tr2) / len(tr)))
        m2 = _fit_models(tr2, Xtr2, lgb_iter=lgb_iter)
        Xte, _, _ = design(te, cols2, stats2)
        out["X"]["train"], out["data"]["train"] = Xtr2, tr2
    else:
        m2 = m1
        Xte, _, _ = design(te, cols, stats)
        out["X"]["train"], out["data"]["train"] = Xtr, tr
    out.update({"final": m2, "cph": m2["cph"], "rsf": m2["rsf"], "gbm": m2["gbm"]})
    out["X"]["test"], out["data"]["test"] = Xte, te
    out["models_for"] = {"val": m1, "test": m2}
    return out


def risk_scores(fit: dict, split: str) -> dict:
    """Higher score = faster restoration (higher hazard of being restored)."""
    m, X = fit["models_for"][split], fit["X"][split]
    return {"Cox PH": m["cph"].predict_partial_hazard(X).to_numpy(),
            "Random survival forest": m["rsf"].predict(X.to_numpy()),
            "LightGBM (log hours)": -m["gbm"].predict(X)}


def predicted_hours(fit: dict, split: str) -> np.ndarray:
    """LightGBM median restoration time in hours (median residual added back)."""
    m = fit["models_for"][split]
    return np.exp(m["gbm"].predict(fit["X"][split]) + np.median(fit["resid"]))


def survival_curves(fit: dict, split: str, times: np.ndarray) -> dict:
    m, X = fit["models_for"][split], fit["X"][split]
    mu = m["gbm"].predict(X)
    r = fit["resid"]
    # S(t|x) = P(log T > log t) = share of residuals above log t - mu(x)
    gbm = 1 - np.searchsorted(r, np.log(times)[None, :] - mu[:, None], side="right") / len(r)
    return {"Cox PH": m["cph"].predict_survival_function(X, times=times).T.to_numpy(),
            "Random survival forest": np.vstack([fn(times) for fn in m["rsf"].predict_survival_function(X.to_numpy())]),
            "LightGBM (log hours)": gbm}


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
    surv = survival_curves(fit, split, times)
    hours = predicted_hours(fit, split)
    rows = [{"model": "Random (no skill)", "C-index": 0.5}]
    for name, sc in scores.items():
        r = {"model": name, "C-index": c_index(d, sc),
             "Integrated Brier": integrated_brier_score(y_tr, y_te, surv[name], times)}
        if name.startswith("LightGBM"):
            obs = d[DURATION].to_numpy()
            r["median abs error (h)"] = np.median(np.abs(hours - obs))
            r["within 2x of actual"] = np.mean((hours / obs <= 2) & (obs / hours <= 2))
        rows.append(r)
    return pd.DataFrame(rows).set_index("model").round(3)


def bootstrap_ci(fit: dict, split: str = "test", n_boot: int = 200, seed: int = SEED) -> pd.DataFrame:
    """95% C-index intervals from a cluster bootstrap over storm episodes."""
    d = fit["data"][split]
    scores = risk_scores(fit, split)
    codes, uniq = pd.factorize(d["episode_id"])
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    T, E = d[DURATION].to_numpy(), d[EVENT].to_numpy()
    rng = np.random.default_rng(seed)
    samp = {k: [] for k in scores}
    for _ in range(n_boot):
        ii = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for k, sc in scores.items():
            samp[k].append(concordance_index(T[ii], -sc[ii], E[ii]))
    return pd.DataFrame({k: {"C-index 95% CI": f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"}
                         for k, v in samp.items()}).T


def hyperparameter_table() -> pd.DataFrame:
    rows = [("Cox PH", "penalizer", COX_PENALIZER), ("Cox PH", "l1_ratio", COX_L1),
            ("Random survival forest", "n_estimators", RSF_TREES)]
    rows += [("Random survival forest", k, v) for k, v in RSF_PARAMS.items()]
    rows += [("LightGBM (log hours)", k, v) for k, v in LGB_SURV.items()]
    rows += [("LightGBM (log hours)", "n_estimators", "early stopping on 2020 (patience 100), scaled for refit")]
    return pd.DataFrame(rows, columns=["model", "hyperparameter", "value"])
