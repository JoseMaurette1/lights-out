"""Hyperparameter search with rolling-origin validation (Optuna, TPE sampler).

Each candidate is scored on two folds and the scores are averaged:
    fold 2019: train 2014-2018, validate on 2019
    fold 2020: train 2014-2019, validate on 2020
History (target-encoded) features are recomputed inside every fold from that
fold's training years only. The 2021-2022 test years are never looked at here.
Averaging two years makes the choice less sensitive to one unusual year.

The winners are copied into models_severity.py / models_survival.py; every
trial is written to reports/tuning_*_trials.csv for the report.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from . import config as C
from . import models_severity as MSEV
from . import models_survival as MSURV
from .features import CATEGORICAL_FEATURES, with_history

SEED = 42
FOLD_YEARS = [2019, 2020]
optuna.logging.set_verbosity(optuna.logging.WARNING)

# search ranges, kept here so the report table and the search cannot drift apart
SPACES = {
    "LightGBM (severity)": {
        "learning_rate": "0.01-0.1 (log)", "num_leaves": "15-255 (log)",
        "min_child_samples": "10-300 (log)", "subsample": "0.5-1.0",
        "colsample_bytree": "0.3-1.0", "reg_lambda": "0.001-30 (log)",
        "min_split_gain": "0-0.5", "cat_smooth": "1-50 (log)",
        "class_weight (classifier)": "none, sqrt_balanced, balanced",
        "n_estimators": "early stopping on the fold's validation year (patience 100)",
    },
    "Random forest": {
        "max_features": "sqrt, 0.2, 0.3, 0.5, 0.7", "min_samples_leaf": "1-60 (log)",
        "max_depth": "None, 12, 20, 30", "max_samples": "0.3-1.0", "n_estimators": "150 in search, 300 final",
    },
    "Ridge / logistic": {"alpha (ridge)": "0.01-1000 (grid)", "C (logistic)": "0.001-100 (grid)"},
    "Random survival forest": {
        "min_samples_leaf": "5-200 (log)", "max_features": "sqrt, 0.2, 0.33, 0.5",
        "max_depth": "None, 10, 16, 24", "max_samples": "0.3-1.0", "n_estimators": "100",
    },
    "LightGBM (log hours)": {
        "learning_rate": "0.01-0.1 (log)", "num_leaves": "15-255 (log)",
        "min_child_samples": "10-300 (log)", "subsample": "0.5-1.0",
        "colsample_bytree": "0.3-1.0", "reg_lambda": "0.001-30 (log)", "min_split_gain": "0-0.5",
        "n_estimators": "early stopping on the fold's validation year (patience 100)",
    },
    "Cox PH": {"penalizer": "0.0001-1 (grid)", "l1_ratio": "0, 0.5"},
}


def folds(df: pd.DataFrame) -> list:
    """[(train, val), ...] with history features recomputed per fold."""
    out = []
    for y in FOLD_YEARS:
        d = with_history(df, list(range(C.TRAIN_YEARS[0], y)))
        out.append((d[d["year"] < y].reset_index(drop=True), d[d["year"] == y].reset_index(drop=True)))
    return out


def _lgb_space(t: optuna.Trial) -> dict:
    return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                min_split_gain=t.suggest_float("min_split_gain", 0, 0.5))


def _rf_space(t: optuna.Trial) -> dict:
    return dict(max_features=t.suggest_categorical("max_features", ["sqrt", 0.2, 0.3, 0.5, 0.7]),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 60, log=True),
                max_depth=t.suggest_categorical("max_depth", [None, 12, 20, 30]),
                max_samples=t.suggest_float("max_samples", 0.3, 1.0))


LGB_FIXED = dict(n_estimators=5000, subsample_freq=1, random_state=SEED, verbose=-1)


def tune_severity(df: pd.DataFrame, model: str, n_trials: int = 30, fold_data=None) -> optuna.Study:
    """model in {lgb_reg, lgb_clf, rf_reg, rf_clf}. Regression minimizes RMSE (log);
    classification maximizes macro-F1 (accuracy is stored alongside)."""
    fd = fold_data or folds(df)
    prepared = []
    for tr, va in fd:
        Xtr = MSEV._lgb_frame(tr)
        Xva = MSEV._lgb_frame(va, {c: Xtr[c].cat.categories for c in CATEGORICAL_FEATURES})
        prepared.append((tr, va, Xtr, Xva))
    stop = [lgb.early_stopping(100, verbose=False)]

    def objective(t):
        if model.startswith("lgb"):
            p = dict(**_lgb_space(t), cat_smooth=t.suggest_float("cat_smooth", 1, 50, log=True), **LGB_FIXED)
            cw = t.suggest_categorical("class_weight", ["none", "sqrt_balanced", "balanced"]) if model == "lgb_clf" else None
        else:
            p = dict(**_rf_space(t), n_estimators=150, n_jobs=-1, random_state=SEED)
        scores, accs, iters = [], [], []
        for tr, va, Xtr, Xva in prepared:
            if model == "lgb_reg":
                m = lgb.LGBMRegressor(**p).fit(Xtr, tr[MSEV.REG_TARGET], eval_set=[(Xva, va[MSEV.REG_TARGET])], callbacks=stop)
                scores.append(mean_squared_error(va[MSEV.REG_TARGET], m.predict(Xva)) ** 0.5); iters.append(m.best_iteration_)
            elif model == "rf_reg":
                m = make_pipeline(MSEV._preprocessor(False), RandomForestRegressor(**p)).fit(tr, tr[MSEV.REG_TARGET])
                scores.append(mean_squared_error(va[MSEV.REG_TARGET], m.predict(va)) ** 0.5)
            else:
                yc, yv = tr[MSEV.CLF_TARGET].astype(str), va[MSEV.CLF_TARGET].astype(str)
                if model == "lgb_clf":
                    m = lgb.LGBMClassifier(class_weight=MSEV.class_weights(yc, cw), **p).fit(
                        Xtr, yc, eval_set=[(Xva, yv)], callbacks=stop)
                    pred = m.predict(Xva); iters.append(m.best_iteration_)
                else:
                    m = make_pipeline(MSEV._preprocessor(False),
                                      RandomForestClassifier(class_weight="balanced_subsample", **p)).fit(tr, yc)
                    pred = m.predict(va)
                scores.append(f1_score(yv, pred, average="macro")); accs.append(accuracy_score(yv, pred))
            t.report(float(np.mean(scores)), len(scores))
        t.set_user_attr("per_fold", [round(s, 4) for s in scores])
        if accs:
            t.set_user_attr("accuracy", round(float(np.mean(accs)), 4))
        if iters:
            t.set_user_attr("n_estimators", [int(i) for i in iters])
        return float(np.mean(scores))

    return _run(objective, model, n_trials, "minimize" if model.endswith("reg") else "maximize")


def tune_linear(df: pd.DataFrame, fold_data=None) -> pd.DataFrame:
    fd = fold_data or folds(df)
    rows = []
    for a in [0.01, 0.1, 1, 10, 100, 1000]:
        s = [mean_squared_error(va[MSEV.REG_TARGET], make_pipeline(MSEV._preprocessor(True), Ridge(alpha=a))
                                .fit(tr, tr[MSEV.REG_TARGET]).predict(va)) ** 0.5 for tr, va in fd]
        rows.append({"model": "Ridge", "param": "alpha", "value": a, "val score (RMSE)": np.mean(s)})
    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
        s = [f1_score(va[MSEV.CLF_TARGET].astype(str), make_pipeline(
            MSEV._preprocessor(True), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
            .fit(tr, tr[MSEV.CLF_TARGET].astype(str)).predict(va), average="macro") for tr, va in fd]
        rows.append({"model": "Logistic", "param": "C", "value": c, "val score (macro F1)": np.mean(s)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.REPORTS / "tuning_linear_grid.csv", index=False)
    return out


def _surv_folds(fd):
    out = []
    for tr, va in fd:
        tr, va = MSURV.survival_frame(tr), MSURV.survival_frame(va)
        Xtr, cols, stats = MSURV.design(tr)
        Xva, _, _ = MSURV.design(va, cols, stats)
        out.append((tr, va, Xtr, Xva))
    return out


def tune_rsf(df: pd.DataFrame, n_trials: int = 15, fold_data=None) -> optuna.Study:
    """Random survival forest on all training rows; maximizes validation C-index."""
    prepared = _surv_folds(fold_data or folds(df))

    def objective(t):
        p = dict(min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 200, log=True),
                 max_features=t.suggest_categorical("max_features", ["sqrt", 0.2, 0.33, 0.5]),
                 max_depth=t.suggest_categorical("max_depth", [None, 10, 16, 24]),
                 max_samples=t.suggest_float("max_samples", 0.3, 1.0))
        scores = []
        for tr, va, Xtr, Xva in prepared:
            m = RandomSurvivalForest(n_estimators=100, low_memory=True, n_jobs=-1, random_state=SEED, **p)
            m.fit(Xtr.to_numpy(), Surv.from_arrays(tr[MSURV.EVENT].astype(bool), tr[MSURV.DURATION]))
            scores.append(MSURV.c_index(va, m.predict(Xva.to_numpy())))
        t.set_user_attr("per_fold", [round(s, 4) for s in scores])
        return float(np.mean(scores))

    return _run(objective, "rsf", n_trials, "maximize")


def tune_lgb_surv(df: pd.DataFrame, n_trials: int = 30, fold_data=None) -> optuna.Study:
    """LightGBM on log restoration hours; maximizes validation C-index."""
    prepared = _surv_folds(fold_data or folds(df))
    stop = [lgb.early_stopping(100, verbose=False)]

    def objective(t):
        p = dict(**_lgb_space(t), **LGB_FIXED)
        scores, iters = [], []
        for tr, va, Xtr, Xva in prepared:
            m = lgb.LGBMRegressor(**p).fit(Xtr, np.log(tr[MSURV.DURATION]),
                                           eval_set=[(Xva, np.log(va[MSURV.DURATION]))], callbacks=stop)
            scores.append(MSURV.c_index(va, -m.predict(Xva))); iters.append(int(m.best_iteration_))
        t.set_user_attr("per_fold", [round(s, 4) for s in scores]); t.set_user_attr("n_estimators", iters)
        return float(np.mean(scores))

    return _run(objective, "lgb_surv", n_trials, "maximize")


def tune_cox(df: pd.DataFrame, fold_data=None) -> pd.DataFrame:
    """Small grid: Cox fits are quick and the response is smooth."""
    prepared = _surv_folds(fold_data or folds(df))
    rows = []
    for pen in [1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]:
        for l1 in [0.0, 0.5]:
            s = []
            for tr, va, Xtr, Xva in prepared:
                c = CoxPHFitter(penalizer=pen, l1_ratio=l1).fit(
                    pd.concat([Xtr, tr[[MSURV.DURATION, MSURV.EVENT]]], axis=1), MSURV.DURATION, MSURV.EVENT)
                s.append(MSURV.c_index(va, c.predict_partial_hazard(Xva).to_numpy()))
            rows.append({"penalizer": pen, "l1_ratio": l1, "val C-index": np.mean(s)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.REPORTS / "tuning_cox_grid.csv", index=False)
    return out


# Settings in use before this search; tried first so tuning can only improve on them
WARM_START = {
    "lgb_clf": dict(learning_rate=0.0206, num_leaves=20, min_child_samples=83, subsample=0.66,
                    colsample_bytree=0.47, reg_lambda=0.0167, min_split_gain=0.076, cat_smooth=13.1,
                    class_weight="sqrt_balanced"),
    "rf_reg": dict(max_features=0.5, min_samples_leaf=6, max_depth=20, max_samples=0.35),
    "rf_clf": dict(max_features=0.2, min_samples_leaf=7, max_depth=None, max_samples=0.78),
    "rsf": dict(min_samples_leaf=7, max_features=0.33, max_depth=None, max_samples=0.43),
}
# Optional: set LIGHTSOUT_OPTUNA_DB=sqlite:///path.db to save every trial and resume after a crash
STORAGE = os.environ.get("LIGHTSOUT_OPTUNA_DB")


def _run(objective, name: str, n_trials: int, direction: str) -> optuna.Study:
    study = optuna.create_study(direction=direction, study_name=name, storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done == 0 and name in WARM_START:
        study.enqueue_trial(WARM_START[name])
    t0 = time.time()
    study.optimize(objective, n_trials=max(0, n_trials - done), callbacks=[lambda s, t: print(
        f"[{name}] trial {t.number:>2}  {t.value:.4f}  best {s.best_value:.4f}  {time.time() - t0:.0f}s",
        flush=True)])
    study.trials_dataframe().to_csv(C.REPORTS / f"tuning_{name}_trials.csv", index=False)
    return study


def summary(studies: dict) -> pd.DataFrame:
    """One row per model: mean validation score over the folds and the chosen hyperparameters."""
    rows = []
    for name, st in studies.items():
        b = st.best_trial
        rows.append({"search": name, "trials": len(st.trials), "best val score (mean of 2019, 2020)": round(b.value, 4),
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.params.items()},
                     **b.user_attrs})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # python -m lightsout.tuning  (long: runs every search)
    import json, sys
    df = pd.read_parquet(C.PROCESSED / "model_table.parquet")
    fd = folds(df)
    N = {"lgb_reg": 40, "lgb_clf": 25, "rf_reg": 20, "rf_clf": 20, "rsf": 15, "lgb_surv": 30}
    which = sys.argv[1:] or ["linear", "cox", "lgb_surv", "lgb_reg", "lgb_clf", "rf_reg", "rf_clf", "rsf"]
    studies = {}
    for w in which:
        if w == "linear":
            print(tune_linear(df, fd).to_string(), flush=True)
        elif w == "cox":
            print(tune_cox(df, fd).to_string(), flush=True)
        elif w == "rsf":
            studies[w] = tune_rsf(df, N[w], fd)
        elif w == "lgb_surv":
            studies[w] = tune_lgb_surv(df, N[w], fd)
        else:
            studies[w] = tune_severity(df, w, N[w], fd)
        if w in studies:
            b = studies[w].best_trial
            print(f"BEST {w}: {b.value:.4f} {json.dumps(b.params, default=str)} {json.dumps(b.user_attrs)}", flush=True)
            summary({w: studies[w]}).to_csv(C.REPORTS / f"tuning_best_{w}.csv", index=False)
