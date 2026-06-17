"""Model training, hyperparameter search, calibration and evaluation.

Upgrades over a basic baseline:
  * time-aware hyperparameter search (RandomizedSearchCV + TimeSeriesSplit)
  * probability calibration (Platt scaling) with a reliability curve
  * permutation importance (model-agnostic, no leakage)
  * rich metrics: accuracy, ROC-AUC, log-loss, Brier + a Brier skill score
"""
from __future__ import annotations
import json, logging, os
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             brier_score_loss, roc_curve)
import xgboost as xgb

from features import FEATURE_COLS

log = logging.getLogger("nba.model")
REPORTS = "reports"


@dataclass
class Metrics:
    accuracy: float
    roc_auc: float
    log_loss: float
    brier: float
    brier_skill: float  # vs. always-predict-base-rate


def _metrics(y, p) -> Metrics:
    base = np.full_like(p, y.mean(), dtype=float)
    bs, bs_ref = brier_score_loss(y, p), brier_score_loss(y, base)
    return Metrics(
        accuracy=round(float(accuracy_score(y, p > 0.5)), 4),
        roc_auc=round(float(roc_auc_score(y, p)), 4),
        log_loss=round(float(log_loss(y, p)), 4),
        brier=round(float(bs), 4),
        brier_skill=round(float(1 - bs / bs_ref), 4),
    )


def time_split(df: pd.DataFrame, test_season: int | None = None):
    test_season = df.season.max() if test_season is None else test_season
    return df[df.season < test_season], df[df.season == test_season]


def _tuned_xgb(Xtr, ytr, n_iter=20, seed=7):
    """Time-aware randomized search for XGBoost hyperparameters."""
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    base = xgb.XGBClassifier(eval_metric="logloss", random_state=seed)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=4), random_state=seed, n_jobs=-1)
    search.fit(Xtr, ytr)
    log.info("best xgb params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_


def train_and_evaluate(features_csv="data/features.csv", tune=True) -> dict:
    os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(features_csv, parse_dates=["date"])
    train, test = time_split(df)
    Xtr, ytr = train[FEATURE_COLS], train.home_win
    Xte, yte = test[FEATURE_COLS], test.home_win

    logit = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        method="sigmoid", cv=5).fit(Xtr, ytr)

    if tune:
        xgb_model, best_params = _tuned_xgb(Xtr, ytr)
    else:
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, eval_metric="logloss", random_state=7).fit(Xtr, ytr)
        best_params = {}

    models = {"logistic_regression": logit, "xgboost": xgb_model}
    preds, metrics = {}, {}
    for name, m in models.items():
        p = m.predict_proba(Xte)[:, 1]
        preds[name] = p
        metrics[name] = asdict(_metrics(yte, p))

    # permutation importance on the tree model (model-agnostic, test set)
    perm = permutation_importance(xgb_model, Xte, yte, n_repeats=20,
                                  random_state=7, scoring="roc_auc")
    importance = (pd.Series(perm.importances_mean, index=FEATURE_COLS)
                  .sort_values())

    _plots(yte, preds, metrics, importance)
    summary = dict(
        baseline=dict(n_train=int(len(train)), n_test=int(len(test)),
                      home_pick_accuracy=round(float((yte == 1).mean()), 4)),
        best_xgb_params=best_params,
        models=metrics,
        permutation_importance={k: round(float(v), 4)
                                for k, v in importance.items()},
    )
    import joblib
    joblib.dump(models["logistic_regression"], f"{REPORTS}/model.joblib")
    with open(f"{REPORTS}/metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    # persist holdout predictions for the backtest
    out = test[["game_id", "season", "date", "home_win"] + FEATURE_COLS].copy()
    out["p_model"] = preds["logistic_regression"]
    # A calibrated Elo-only model stands in for an efficient "market" price, so
    # the backtest's edge reflects genuine extra signal (form, rest), not a
    # miscalibrated baseline.
    market = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        method="sigmoid", cv=5).fit(Xtr[["elo_diff"]], ytr)
    out["p_market"] = market.predict_proba(Xte[["elo_diff"]])[:, 1]
    out.to_csv(f"{REPORTS}/holdout_predictions.csv", index=False)
    return summary


def _plots(yte, preds, metrics, importance):
    plt.figure(figsize=(6, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — home-win")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{REPORTS}/roc.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 5))
    for name, p in preds.items():
        fp, mp = calibration_curve(yte, p, n_bins=10)
        plt.plot(mp, fp, "o-", label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{REPORTS}/calibration.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 4)); importance.plot.barh(color="#2b6cb0")
    plt.title("Permutation importance (ROC-AUC drop)"); plt.tight_layout()
    plt.savefig(f"{REPORTS}/feature_importance.png", dpi=120); plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(train_and_evaluate(), indent=2))
