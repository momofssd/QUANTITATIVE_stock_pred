"""Exercise package migration and the full pipeline without external providers."""

import importlib
import json
import pickle
from functools import partial

import joblib
import numpy as np
import pandas as pd
import pytest

from src import predict
from src.data_fetching import MarketDataBundle
from src.data_modeling import build_features, model_matrix, short_term_model_matrix, save_signal_outputs
from src.machine_learning import predict_probabilities, save_trained_models
from src.machine_learning.train import SoftVoteEnsemble


@pytest.mark.parametrize(
    "old,new",
    [
        ("data_loader", "data_fetching.data_loader"),
        ("indicators", "data_modeling.indicators"),
        ("features", "data_modeling.features"),
        ("labels", "data_modeling.labels"),
        ("decision_engine", "data_modeling.decision_engine"),
        ("short_term_trade", "data_modeling.short_term_trade"),
        ("json_exporter", "data_modeling.json_exporter"),
        ("train", "machine_learning.train"),
        ("five_day_forecast", "machine_learning.five_day_forecast"),
        ("gpt_interpreter", "ai_interpretation.gemini"),
    ],
)
def test_legacy_imports_resolve_to_implementation(old, new):
    assert importlib.import_module(f"src.{old}") is importlib.import_module(f"src.{new}")


def test_legacy_serialized_ensemble_class_resolves():
    # Protocol 0 GLOBAL references the path embedded in older saved ensembles.
    assert pickle.loads(b"csrc.train\nSoftVoteEnsemble\n.") is SoftVoteEnsemble


@pytest.fixture
def market_bundle():
    index = pd.date_range("2020-01-01", periods=620, freq="B", name="Date")
    steps = np.arange(len(index))

    def history(base, drift, amplitude):
        close = base + drift * steps + amplitude * np.sin(steps / 18) + np.sin(steps / 3)
        return pd.DataFrame(
            {
                "Open": close - 0.3,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Adj Close": close,
                "Volume": 1_000_000 + 100_000 * np.cos(steps / 7),
            },
            index=index,
        )

    return MarketDataBundle(
        ticker="TEST",
        ticker_data=history(100, 0.04, 12),
        benchmark_primary="QQQ",
        benchmark_primary_data=history(200, 0.05, 5),
        benchmark_secondary="SPY",
        benchmark_secondary_data=history(300, 0.05, 7),
        vix_data=history(20, 0, 2),
        treasury_yield_data=history(4, 0, 0.1),
        oil_data=history(70, 0.01, 3),
        warnings=[],
    )


@pytest.mark.parametrize("save_artifacts", [False, True])
def test_quant_pipeline_and_saved_model_round_trip(monkeypatch, tmp_path, market_bundle, save_artifacts):
    monkeypatch.setattr(predict, "load_market_data", lambda *args: market_bundle)
    monkeypatch.setattr(predict, "ensure_directories", lambda: None)
    monkeypatch.setattr(predict, "save_trained_models", partial(save_trained_models, model_dir=tmp_path / "models"))
    monkeypatch.setattr(predict, "save_signal_outputs", partial(save_signal_outputs, output_dir=tmp_path / "outputs"))

    signal, report = predict.generate_quant_signal(
        "TEST", model_name="extratrees", save_outputs=save_artifacts, save_models=save_artifacts,
    )

    assert signal["ticker"] == "TEST"
    assert signal["as_of_date"] == str(market_bundle.ticker_data.index[-1].date())
    assert signal["schema_version"] == "2.2"
    assert len(signal["chart_data"]["points"]) == 180
    assert signal["model_metadata"]["purged_rows"] == 60
    assert signal["five_day_forecast"]["validation"]["purged_rows"] == 5
    assert signal["short_term_trade"]["action"]
    assert len(signal["five_day_forecast"]["session_forecasts"]) == 5
    assert signal["five_day_forecast"]["price_action_plan"]["levels"]["support"] > 0
    assert "cci_20" in signal["technical_indicators"]
    assert "macro avg" in report
    json.dumps(signal, allow_nan=False)

    if not save_artifacts:
        assert list(tmp_path.iterdir()) == []
        return

    assert json.loads((tmp_path / "outputs" / "latest_signal.json").read_text()) == signal
    assert "TEST Long-Term Signal" in (tmp_path / "outputs" / "latest_signal.md").read_text()
    paths = list((tmp_path / "models").glob("*.joblib"))
    assert len(paths) == 2
    long_model = joblib.load(next(path for path in paths if "extratreesclassifier" in path.name))
    short_models = joblib.load(next(path for path in paths if "five_day_forecast" in path.name))
    features = build_features(
        market_bundle.ticker_data, market_bundle.benchmark_primary_data,
        market_bundle.benchmark_secondary_data, market_bundle.vix_data,
        market_bundle.treasury_yield_data, market_bundle.oil_data,
    )
    latest_X = model_matrix(features).tail(1)
    probabilities = predict_probabilities(long_model, latest_X)
    exported = signal["current_market"]["model_probabilities"]
    for label, field in [("bullish", "bull_trend"), ("sideways", "sideways"), ("bearish", "bear_trend")]:
        assert probabilities[label] == pytest.approx(exported[field], abs=0.0001)
    assert {"direction_model", "return_model", "conditional_return_models", "feature_columns", "interval_radius", "validation", "live_fit", "calibration_residuals", "session_models"} == set(short_models)
    assert set(short_models["session_models"]) == {1, 2, 3, 4}
    short_X = short_term_model_matrix(features).tail(1)
    assert list(short_X.columns) == short_models["feature_columns"]
    assert short_models["direction_model"].predict_proba(short_X)[0, 1] == pytest.approx(
        signal["five_day_forecast"]["probability_of_gain"], abs=0.0001,
    )
    assert short_models["validation"] == signal["five_day_forecast"]["validation"]
