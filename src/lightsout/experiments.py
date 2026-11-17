"""Validation-only experiments: which data and which models help?

Everything here trains on 2014-2019 and scores on 2020. The 2021-2022 test years are not
touched. Learning rate is raised to 0.05 so each fit takes minutes; the final models refit
with the tuned settings.

  python -m lightsout.experiments data     # feature-set ablation (LightGBM)
  python -m lightsout.experiments models   # model types on the chosen feature set
Results go to reports/experiments_*.csv.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_squared_error, roc_auc_score)

from . import config as C
from . import models_severity as S
from . import models_survival as V
from .extra_data import EXTRA_FEATURES
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

BASE = [c for c in NUMERIC_FEATURES if c not in EXTRA_FEATURES]
WX = [c for c in EXTRA_FEATURES if c.startswith("wx_")]
ASOS = [c for c in EXTRA_FEATURES if c.startswith("asos_")]
EIA = [c for c in EXTRA_FEATURES if c.startswith("eia_")]
SETS = {"base": BASE, "+weather (nClimGrid)": BASE + WX, "+wind (ASOS)": BASE + ASOS,
        "+utility (EIA)": BASE + EIA, "+weather+wind": BASE + WX + ASOS, "+all": BASE + WX + ASOS + EIA}
FAST = dict(learning_rate=0.05, n_estimators=4000, subsample_freq=1, random_state=42, verbose=-1)
STOP = [lgb.early_stopping(100, verbose=False)]


def frame(d, num, cats=None):
    X = d[num + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=None if cats is None else cats[c])
    return X


def clf_scores(y, P, classes) -> dict:
    j = list(classes).index("high")
    yh = (np.asarray(y) == "high")
    pred = np.asarray(classes)[P.argmax(1)]
    return {"macro F1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
            "AUC high": roc_auc_score(yh, P[:, j]), "AP high": average_precision_score(yh, P[:, j]),
            "recall high": (pred[yh] == "high").mean()}


def severity_run(tr, va, num, params=None, weights="sqrt_balanced"):
    p = {**{k: v for k, v in S.LGB_CLF.items() if k != "learning_rate"}, **FAST, **(params or {})}
    Xtr = frame(tr, num); cats = {c: Xtr[c].cat.categories for c in CATEGORICAL_FEATURES}
    Xva = frame(va, num, cats)
    yc, yv = tr[S.CLF_TARGET].astype(str), va[S.CLF_TARGET].astype(str)
    clf = lgb.LGBMClassifier(class_weight=S.class_weights(yc, weights), **p).fit(
        Xtr, yc, eval_set=[(Xva, yv)], callbacks=STOP)
    pr = {**{k: v for k, v in S.LGB_REG.items() if k != "learning_rate"}, **FAST}
    reg = lgb.LGBMRegressor(**pr).fit(Xtr, tr[S.REG_TARGET], eval_set=[(Xva, va[S.REG_TARGET])], callbacks=STOP)
    out = clf_scores(yv, clf.predict_proba(Xva), clf.classes_)
    out["RMSE (log)"] = mean_squared_error(va[S.REG_TARGET], reg.predict(Xva)) ** 0.5
    return out, clf, reg, Xva


def survival_run(df, num):
    s = V.survival_frame(df)
    tr, va = s[s.split == "train"], s[s.split == "val"]
    Xtr = frame(tr, num + ["log_peak_pct"]); cats = {c: Xtr[c].cat.categories for c in CATEGORICAL_FEATURES}
    Xva = frame(va, num + ["log_peak_pct"], cats)
    p = {**{k: v for k, v in V.LGB_SURV.items() if k != "learning_rate"}, **FAST}
    m = lgb.LGBMRegressor(**p).fit(Xtr, np.log(tr[V.DURATION]), eval_set=[(Xva, np.log(va[V.DURATION]))],
                                   callbacks=STOP)
    return V.c_index(va.reset_index(drop=True), -m.predict(Xva))


def data_ablation(df, only=None) -> pd.DataFrame:
    tr, va, _ = S.splits(df)
    rows = []
    for name, num in SETS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        sev, *_ = severity_run(tr, va, num)
        rows.append({"features": name, "n features": len(num) + len(CATEGORICAL_FEATURES), **sev,
                     "restoration C-index": survival_run(df, num), "minutes": (time.time() - t0) / 60})
        print(pd.DataFrame(rows).round(4).to_string(), flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.REPORTS / ("experiments_data.csv" if not only else "experiments_data_extra.csv"), index=False)
    return out


def model_comparison(df, num) -> pd.DataFrame:
    """LightGBM vs CatBoost vs XGBoost vs their average, severity classes on 2020."""
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    tr, va, _ = S.splits(df)
    yc, yv = tr[S.CLF_TARGET].astype(str), va[S.CLF_TARGET].astype(str)
    rows, probs = [], {}
    sev, clf, _, Xva = severity_run(tr, va, num)
    probs["LightGBM"], classes = clf.predict_proba(Xva), clf.classes_
    rows.append({"model": "LightGBM", **sev})
    print(rows[-1], flush=True)

    cw = S.class_weights(yc, "sqrt_balanced")
    Xtr_c, Xva_c = tr[num + CATEGORICAL_FEATURES].copy(), va[num + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        Xtr_c[c], Xva_c[c] = Xtr_c[c].astype(str), Xva_c[c].astype(str)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=7, l2_leaf_reg=5, loss_function="MultiClass",
                            class_weights=cw, cat_features=CATEGORICAL_FEATURES, random_seed=42, verbose=0,
                            od_type="Iter", od_wait=100, thread_count=-1)
    cb.fit(Xtr_c, yc, eval_set=(Xva_c, yv), use_best_model=True)
    P = cb.predict_proba(Xva_c); P = P[:, [list(cb.classes_).index(k) for k in classes]]
    probs["CatBoost"] = P
    rows.append({"model": "CatBoost", **clf_scores(yv, P, classes)}); print(rows[-1], flush=True)

    Xtr_x, Xva_x = frame(tr, num), None
    cats = {c: Xtr_x[c].cat.categories for c in CATEGORICAL_FEATURES}; Xva_x = frame(va, num, cats)
    lab = {k: i for i, k in enumerate(classes)}
    sw = yc.map(cw).to_numpy()
    xg = XGBClassifier(n_estimators=4000, learning_rate=0.05, max_depth=7, subsample=0.7, colsample_bytree=0.6,
                       min_child_weight=5, reg_lambda=5, tree_method="hist", enable_categorical=True,
                       early_stopping_rounds=100, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xg.fit(Xtr_x, yc.map(lab), sample_weight=sw, eval_set=[(Xva_x, yv.map(lab))], verbose=False)
    probs["XGBoost"] = xg.predict_proba(Xva_x)
    rows.append({"model": "XGBoost", **clf_scores(yv, probs["XGBoost"], classes)}); print(rows[-1], flush=True)

    for combo in [("LightGBM", "CatBoost"), ("LightGBM", "CatBoost", "XGBoost")]:
        P = np.mean([probs[k] for k in combo], axis=0)
        rows.append({"model": " + ".join(combo) + " (average)", **clf_scores(yv, P, classes)}); print(rows[-1], flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.REPORTS / "experiments_models.csv", index=False)
    return out


if __name__ == "__main__":
    df = pd.read_parquet(C.PROCESSED / "model_table.parquet")
    what = sys.argv[1] if len(sys.argv) > 1 else "data"
    if what == "data":
        print(data_ablation(df, sys.argv[2:] or None).to_string())
    else:
        print(model_comparison(df, SETS[sys.argv[2] if len(sys.argv) > 2 else "+all"]).to_string())
