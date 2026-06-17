import os, tempfile
import pandas as pd
from simulate import simulate_seasons
from features import build_features
import model as M

def test_pipeline_beats_coinflip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data"); os.makedirs("reports", exist_ok=True)
    build_features(simulate_seasons(n_seasons=3, seed=4)).to_csv("data/features.csv", index=False)
    summary = M.train_and_evaluate(tune=False)   # fast: no hyperparameter search
    for name in ("logistic_regression", "xgboost"):
        assert summary["models"][name]["roc_auc"] > 0.5
    assert os.path.exists("reports/model.joblib")
    assert os.path.exists("reports/holdout_predictions.csv")
