"""Outage severity models (Objective 1).

Regression target: log_peak_pct = log(1 + peak % of customers out).
Classification target: severity in {low, medium, high}.
Models: storm-type baseline, linear/logistic, random forest, LightGBM.

Protocol
  1. Hyperparameters are chosen by rolling-origin validation (tuning.py):
     train on the years before 2019 / 2020, score on 2019 / 2020.
  2. Stage 1: fit on 2014-2019, early-stop LightGBM on 2020 -> validation scores.
  3. Stage 2: refit every model on 2014-2020 with the same settings (LightGBM
     keeps its early-stopped number of trees, scaled to the larger data) and
     with history features recomputed from 2014-2020 -> test scores on 2021-2022.
     The test years are never used to choose anything.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, with_history

REG_TARGET = "log_peak_pct"
CLF_TARGET = "severity"
CLASSES = ["low", "medium", "high"]
SEED = 42
FINAL_TRAIN_YEARS = C.TRAIN_YEARS + C.VAL_YEARS

# Chosen by tuning.py (rolling-origin validation on 2019 and 2020; ranges in tuning.SPACES).
# Run notebook 05 to regenerate; reports/tuning_best.csv holds the same values.
RIDGE_ALPHA = 1000.0  # best of 0.01-1000; differences are tiny (RMSE .825-.828), see tuning_linear_grid.csv
LOGIT_C = 10.0
RF_REG = dict(max_features=0.5, min_samples_leaf=6, max_depth=20, max_samples=0.35)
RF_CLF = dict(max_features=0.2, min_samples_leaf=7, max_depth=None, max_samples=0.78)
LGB_REG = dict(learning_rate=0.0111, num_leaves=84, min_child_samples=115, subsample=0.793,
               colsample_bytree=0.519, reg_lambda=0.0016, min_split_gain=0.162, cat_smooth=1.55)
LGB_CLF = dict(learning_rate=0.0217, num_leaves=15, min_child_samples=244, subsample=0.541,
               colsample_bytree=0.439, reg_lambda=10.07, min_split_gain=0.258, cat_smooth=7.17)
LGB_CLF_WEIGHTS = "sqrt_balanced"
RF_TREES = 300


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
    y, p = np.asarray(y).astype(str), np.asarray(p).astype(str)
    rec = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)[1]
    return {"macro F1": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p),
            **{f"recall {c}": r for c, r in zip(CLASSES, rec)}}


def class_weights(y: pd.Series, kind: str) -> dict | str | None:
    """None, "balanced" (inverse class frequency) or its square root, a gentler
    up-weighting of rare classes that costs less accuracy on the common one."""
    if kind in (None, "none"):
        return None
    if kind != "sqrt_balanced":
        return kind
    freq = y.value_counts(normalize=True)
    return {k: float((1 / len(freq) / v) ** 0.5) for k, v in freq.items()}


def decision_weights(P: np.ndarray, y, classes) -> np.ndarray:
    """Per-class multipliers on predicted probabilities that maximize macro-F1
    (grid search; the most common class is fixed at 1). Fit on validation only."""
    y = np.asarray(y).astype(str)
    j0 = int(np.argmax([np.mean(y == c) for c in classes]))
    others = [j for j in range(len(classes)) if j != j0]
    grid = np.round(np.arange(0.5, 4.01, 0.1), 2)
    best, best_w = -1, np.ones(len(classes))
    for a in grid:
        for b in grid:
            w = np.ones(len(classes)); w[others[0]], w[others[1]] = a, b
            f = f1_score(y, np.asarray(classes)[np.argmax(P * w, 1)], average="macro")
            if f > best:
                best, best_w = f, w
    return best_w


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


def _fit_models(tr: pd.DataFrame, va: pd.DataFrame | None, n_iter: dict | None = None) -> dict:
    """Fit every model on `tr`. With `va`, LightGBM early-stops on it; otherwise
    it uses the fixed tree counts in `n_iter`."""
    m = {}
    ytr_r, ytr_c = tr[REG_TARGET], tr[CLF_TARGET].astype(str)
    m["Baseline (storm type)", "reg"] = (tr.groupby("storm_type")[REG_TARGET].mean(), ytr_r.mean())
    m["Baseline (storm type)", "clf"] = (
        tr.groupby("storm_type")[CLF_TARGET].agg(lambda s: s.value_counts().idxmax()).astype(str),
        ytr_c.value_counts().idxmax())
    m["Linear / logistic", "reg"] = make_pipeline(_preprocessor(True), Ridge(alpha=RIDGE_ALPHA)).fit(tr, ytr_r)
    m["Linear / logistic", "clf"] = make_pipeline(
        _preprocessor(True), LogisticRegression(C=LOGIT_C, max_iter=3000, class_weight="balanced")).fit(tr, ytr_c)
    rf_kw = dict(n_estimators=RF_TREES, n_jobs=-1, random_state=SEED)
    m["Random forest", "reg"] = make_pipeline(
        _preprocessor(False), RandomForestRegressor(**RF_REG, **rf_kw)).fit(tr, ytr_r)
    m["Random forest", "clf"] = make_pipeline(
        _preprocessor(False), RandomForestClassifier(class_weight="balanced_subsample", **RF_CLF, **rf_kw)
    ).fit(tr, ytr_c)

    Xtr = _lgb_frame(tr)
    cats = {c: Xtr[c].cat.categories for c in CATEGORICAL_FEATURES}
    kw = dict(subsample_freq=1, random_state=SEED, verbose=-1)
    if va is not None:
        Xva = _lgb_frame(va, cats)
        stop = [lgb.early_stopping(100, verbose=False)]
        lr = lgb.LGBMRegressor(n_estimators=5000, **LGB_REG, **kw).fit(
            Xtr, ytr_r, eval_set=[(Xva, va[REG_TARGET])], callbacks=stop)
        lc = lgb.LGBMClassifier(n_estimators=5000, class_weight=class_weights(ytr_c, LGB_CLF_WEIGHTS),
                                **LGB_CLF, **kw).fit(
            Xtr, ytr_c, eval_set=[(Xva, va[CLF_TARGET].astype(str))], callbacks=stop)
    else:
        lr = lgb.LGBMRegressor(n_estimators=n_iter["reg"], **LGB_REG, **kw).fit(Xtr, ytr_r)
        lc = lgb.LGBMClassifier(n_estimators=n_iter["clf"], class_weight=class_weights(ytr_c, LGB_CLF_WEIGHTS),
                                **LGB_CLF, **kw).fit(Xtr, ytr_c)
    m["LightGBM", "reg"], m["LightGBM", "clf"] = lr, lc
    m["_cats"] = cats
    return m


def _predict(m: dict, d: pd.DataFrame) -> dict:
    X = _lgb_frame(d, m["_cats"])
    out = {}
    for (name, kind), model in ((k, v) for k, v in m.items() if k != "_cats"):
        if name.startswith("Baseline"):
            table, fallback = model
            p = d["storm_type"].map(table)
            out[f"{name}|{kind}"] = (p.fillna(fallback) if kind == "reg" else p.astype(object).fillna(fallback)).to_numpy()
        elif name == "LightGBM":
            out[f"{name}|{kind}"] = model.predict(X)
        else:
            out[f"{name}|{kind}"] = model.predict(d)
    out["_proba"] = m["LightGBM", "clf"].predict_proba(X)
    return out


def fit_all(df: pd.DataFrame, refit: bool = True) -> dict:
    """Stage 1 (2014-2019 -> 2020 validation) and stage 2 (2014-2020 -> 2021-22 test).
    With refit=False the stage-1 models also produce the test predictions."""
    tr, va, te = splits(df)
    m1 = _fit_models(tr, va)
    pv = _predict(m1, va)
    classes = m1["LightGBM", "clf"].classes_
    w = decision_weights(pv["_proba"], va[CLF_TARGET], classes)

    if refit:
        d2 = with_history(df, FINAL_TRAIN_YEARS)
        tr2 = d2[d2["year"].isin(FINAL_TRAIN_YEARS)].reset_index(drop=True)
        te = d2[d2["split"] == "test"].reset_index(drop=True)
        scale = len(tr2) / len(tr)  # more rows -> proportionally more boosting rounds
        n_iter = {k: int(round(m1["LightGBM", k].best_iteration_ * scale)) for k in ("reg", "clf")}
        m2 = _fit_models(tr2, None, n_iter)
    else:
        m2, n_iter = m1, None
    pt = _predict(m2, te)

    pred = {}
    for split, d, p in (("val", va, pv), ("test", te, pt)):
        f = d[["episode_id", "fips"]].copy()
        for k, v in p.items():
            if not k.startswith("_"):
                f[k] = v
        f["LightGBM (F1-tuned cutoffs)|clf"] = classes[np.argmax(p["_proba"] * w, 1)]
        pred[split] = f
    return {"models": {"stage1": m1, "final": m2}, "pred": pred, "proba": {"val": pv["_proba"], "test": pt["_proba"]},
            "classes": classes, "decision_weights": dict(zip(classes, w.round(2))),
            "n_iter": {"stage1": {k: m1["LightGBM", k].best_iteration_ for k in ("reg", "clf")}, "final": n_iter},
            "lgb_frames": {"test": _lgb_frame(te, m2["_cats"])}, "truth": {"val": va, "test": te}}


def results_table(fit: dict, split: str = "test") -> pd.DataFrame:
    truth, pred = fit["truth"][split], fit["pred"][split]
    rows: dict = {}
    for col in pred.columns[2:]:
        name, kind = col.split("|")
        r = rows.setdefault(name, {"model": name})
        if kind == "reg":
            r.update(reg_metrics(truth[REG_TARGET], pred[col]))
        else:
            r.update(clf_metrics(truth[CLF_TARGET], pred[col]))
    return pd.DataFrame(rows.values()).set_index("model").round(3)


def per_class_table(fit: dict, col: str = "LightGBM|clf", split: str = "test") -> pd.DataFrame:
    y = fit["truth"][split][CLF_TARGET].astype(str)
    p, r, f, n = precision_recall_fscore_support(y, fit["pred"][split][col].astype(str), labels=CLASSES)
    return pd.DataFrame({"precision": p, "recall": r, "F1": f, "n": n}, index=CLASSES).round(3)


def bootstrap_ci(fit: dict, split: str = "test", n_boot: int = 500, seed: int = SEED) -> pd.DataFrame:
    """95% intervals from a cluster bootstrap over storm episodes (rows from the
    same storm are correlated, so resampling rows would understate uncertainty)."""
    truth, pred = fit["truth"][split], fit["pred"][split]
    codes, uniq = pd.factorize(truth["episode_id"])
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    lab = {c: i for i, c in enumerate(CLASSES)}
    y_r = truth[REG_TARGET].to_numpy()
    y_c = truth[CLF_TARGET].astype(str).map(lab).to_numpy()
    cols = {c: (c.split("|")[1], pred[c].to_numpy() if c.endswith("reg") else pred[c].astype(str).map(lab).to_numpy())
            for c in pred.columns[2:]}
    rng = np.random.default_rng(seed)
    samples = {c: [] for c in cols}
    for _ in range(n_boot):
        ii = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for c, (kind, p) in cols.items():
            if kind == "reg":
                samples[c].append(mean_squared_error(y_r[ii], p[ii]) ** 0.5)
            else:
                samples[c].append((f1_score(y_c[ii], p[ii], average="macro"), accuracy_score(y_c[ii], p[ii])))
    rows: dict = {}
    for c, s in samples.items():
        name, kind = c.split("|")
        r = rows.setdefault(name, {"model": name})
        s = np.asarray(s)
        fmt = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
        if kind == "reg":
            r["RMSE (log) 95% CI"] = fmt(s)
        else:
            r["macro F1 95% CI"], r["accuracy 95% CI"] = fmt(s[:, 0]), fmt(s[:, 1])
    return pd.DataFrame(rows.values()).set_index("model")


def hyperparameter_table() -> pd.DataFrame:
    """Every setting used by the final severity models (for the report)."""
    rows = [("Baseline (storm type)", "rule", "mean (regression) / most common class per storm type"),
            ("Ridge", "alpha", RIDGE_ALPHA), ("Logistic", "C", LOGIT_C),
            ("Logistic", "class_weight", "balanced"), ("Logistic", "max_iter", 3000),
            ("Random forest (both)", "n_estimators", RF_TREES)]
    rows += [("Random forest regressor", k, v) for k, v in RF_REG.items()]
    rows += [("Random forest classifier", k, v) for k, v in RF_CLF.items()]
    rows += [("Random forest classifier", "class_weight", "balanced_subsample")]
    rows += [("LightGBM regressor", k, v) for k, v in LGB_REG.items()]
    rows += [("LightGBM classifier", k, v) for k, v in LGB_CLF.items()]
    rows += [("LightGBM classifier", "class_weight", LGB_CLF_WEIGHTS),
             ("LightGBM (both)", "n_estimators", "early stopping on 2020 (patience 100), x1.21 for the 2014-2020 refit"),
             ("LightGBM (both)", "subsample_freq", 1)]
    return pd.DataFrame(rows, columns=["model", "hyperparameter", "value"])


def shap_values(fit: dict, n: int = 5000):
    import shap
    X = fit["lgb_frames"]["test"]
    X = X.sample(min(n, len(X)), random_state=SEED)
    expl = shap.TreeExplainer(fit["models"]["final"][("LightGBM", "reg")])
    return expl(X), X
