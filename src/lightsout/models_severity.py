"""Outage severity models (Objective 1).

Regression target: log_peak_pct = log(1 + peak % of customers out).
Classification target: severity in {low, medium, high}.
Models: storm-type baseline, linear/logistic, random forest, LightGBM.
Train on TRAIN_YEARS, tune/early-stop on VAL_YEARS, report on TEST_YEARS.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

REG_TARGET = "log_peak_pct"
CLF_TARGET = "severity"
SEED = 42


def splits(df: pd.DataFrame):
    return (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))


def to_percent(log_pct: np.ndarray) -> np.ndarray:
    return np.expm1(log_pct)


def reg_metrics(y_log, p_log) -> dict:
    return {
        "RMSE (log)": mean_squared_error(y_log, p_log) ** 0.5,
        "MAE (log)": mean_absolute_error(y_log, p_log),
        "MAE (pct pts)": mean_absolute_error(to_percent(y_log), to_percent(p_log)),
    }


def clf_metrics(y, p) -> dict:
    return {"macro F1": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p)}


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([
        ("num", make_pipeline(*num), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_FEATURES),
    ])


def _lgb_frame(df: pd.DataFrame, cats=None) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=None if cats is None else cats[c])
    return X


def fit_all(df: pd.DataFrame) -> dict:
    """Fit every model; return predictions on val/test plus fitted models."""
    tr, va, te = splits(df)
    out = {"models": {}, "pred": {"val": va[["episode_id", "fips"]].copy(),
                                  "test": te[["episode_id", "fips"]].copy()}}

    def record(name, model, kind, pv, pt):
        out["models"][(name, kind)] = model
        out["pred"]["val"][f"{name}|{kind}"] = pv
        out["pred"]["test"][f"{name}|{kind}"] = pt

    # 1) baseline: storm type only
    m = tr.groupby("storm_type")[REG_TARGET].mean()
    g = tr[REG_TARGET].mean()
    record("Baseline (storm type)", m, "reg",
           va["storm_type"].map(m).fillna(g).to_numpy(), te["storm_type"].map(m).fillna(g).to_numpy())
    mc = tr.groupby("storm_type")[CLF_TARGET].agg(lambda s: s.value_counts().idxmax())
    gc = tr[CLF_TARGET].value_counts().idxmax()
    record("Baseline (storm type)", mc, "clf",
           va["storm_type"].map(mc).astype(object).fillna(gc).to_numpy(),
           te["storm_type"].map(mc).astype(object).fillna(gc).to_numpy())

    # 2) linear / logistic
    lin = make_pipeline(_preprocessor(scale=True), Ridge(alpha=1.0)).fit(tr, tr[REG_TARGET])
    record("Linear / logistic", lin, "reg", lin.predict(va), lin.predict(te))
    log = make_pipeline(_preprocessor(scale=True),
                        LogisticRegression(max_iter=2000, class_weight="balanced")
                        ).fit(tr, tr[CLF_TARGET].astype(str))
    record("Linear / logistic", log, "clf", log.predict(va), log.predict(te))

    # 3) random forest
    rf_kw = dict(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=SEED)
    rfr = make_pipeline(_preprocessor(scale=False), RandomForestRegressor(**rf_kw)).fit(tr, tr[REG_TARGET])
    record("Random forest", rfr, "reg", rfr.predict(va), rfr.predict(te))
    rfc = make_pipeline(_preprocessor(scale=False),
                        RandomForestClassifier(class_weight="balanced_subsample", **rf_kw)
                        ).fit(tr, tr[CLF_TARGET].astype(str))
    record("Random forest", rfc, "clf", rfc.predict(va), rfc.predict(te))

    # 4) LightGBM with early stopping on the validation year
    Xtr = _lgb_frame(tr)
    cats = {c: Xtr[c].cat.categories for c in CATEGORICAL_FEATURES}
    Xva, Xte = _lgb_frame(va, cats), _lgb_frame(te, cats)
    lgb_kw = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1)
    stop = [lgb.early_stopping(100, verbose=False)]
    lr = lgb.LGBMRegressor(**lgb_kw).fit(Xtr, tr[REG_TARGET], eval_set=[(Xva, va[REG_TARGET])],
                                         callbacks=stop)
    record("LightGBM", lr, "reg", lr.predict(Xva), lr.predict(Xte))
    lc = lgb.LGBMClassifier(class_weight="balanced", **lgb_kw).fit(
        Xtr, tr[CLF_TARGET].astype(str), eval_set=[(Xva, va[CLF_TARGET].astype(str))],
        callbacks=stop)
    record("LightGBM", lc, "clf", lc.predict(Xva), lc.predict(Xte))
    out["lgb_frames"] = {"train": Xtr, "val": Xva, "test": Xte}
    out["truth"] = {"val": va, "test": te}
    return out


def results_table(fit: dict, split: str = "test") -> pd.DataFrame:
    truth, pred = fit["truth"][split], fit["pred"][split]
    rows = []
    for col in pred.columns[2:]:
        name, kind = col.split("|")
        if kind == "reg":
            rows.append({"model": name, **reg_metrics(truth[REG_TARGET], pred[col])})
        else:
            rows[-1].update(clf_metrics(truth[CLF_TARGET].astype(str), pred[col].astype(str)))
    return pd.DataFrame(rows).set_index("model").round(3)


def shap_values(fit: dict, n: int = 5000):
    import shap
    X = fit["lgb_frames"]["test"]
    X = X.sample(min(n, len(X)), random_state=SEED)
    expl = shap.TreeExplainer(fit["models"][("LightGBM", "reg")])
    return expl(X), X
