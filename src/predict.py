from __future__ import annotations

from datetime import date

import joblib

from .data_loader import load_market_data
from .decision_engine import calculate_decision
from .features import FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS, build_features, model_matrix
from .five_day_forecast import generate_five_day_forecast
from .json_exporter import build_signal_json, save_signal_outputs
from .labels import create_basic_labels, infer_regime
from .short_term_trade import build_short_term_trade_plan
from .train import predict_probabilities, train_classifier
from .utils import MODEL_DIR, ensure_directories


def generate_quant_signal(
    ticker: str,
    start_date: str = "2012-01-01",
    end_date: str | None = None,
    benchmark_primary: str = "QQQ",
    benchmark_secondary: str = "SPY",
    treasury_yield_ticker: str = "^TNX",
    oil_ticker: str = "CL=F",
    horizon_days: int = 60,
    model_name: str = "xgboost",
    save_outputs: bool = True,
    save_models: bool = True,
) -> tuple[dict, str]:
    ensure_directories()
    bundle = load_market_data(
        ticker,
        start_date,
        end_date,
        benchmark_primary,
        benchmark_secondary,
        treasury_yield_ticker,
        oil_ticker,
    )
    features = build_features(
        bundle.ticker_data,
        bundle.benchmark_primary_data,
        bundle.benchmark_secondary_data,
        bundle.vix_data,
        bundle.treasury_yield_data,
        bundle.oil_data,
    )
    labels = create_basic_labels(features, horizon_days=horizon_days)
    X = model_matrix(features)
    training = train_classifier(X, labels, model_name=model_name, purge_gap=horizon_days)

    usable_features = features.loc[X.dropna(how="all").index].dropna(subset=["close"])
    latest_index = usable_features.index[-1]
    latest_X = X.loc[[latest_index]]
    probabilities = predict_probabilities(training.model, latest_X)
    regime = infer_regime(features.loc[latest_index], probabilities)
    decision = calculate_decision(features.loc[:latest_index].dropna(subset=["close"]), regime, probabilities)
    five_day_forecast, five_day_training = generate_five_day_forecast(X, features, latest_index)
    short_term_trade = build_short_term_trade_plan(
        features.loc[:latest_index].dropna(subset=["close"]),
        regime,
        five_day_forecast,
        decision.price_zones,
    )

    signal = build_signal_json(
        ticker=bundle.ticker,
        as_of_date=str(latest_index.date()),
        benchmark_primary=bundle.benchmark_primary,
        benchmark_secondary=bundle.benchmark_secondary,
        treasury_yield_ticker=bundle.treasury_yield_ticker,
        oil_ticker=bundle.oil_ticker,
        features=features.loc[:latest_index],
        regime=regime,
        probabilities=probabilities,
        decision_result=decision,
        five_day_forecast=five_day_forecast,
        short_term_trade=short_term_trade,
        model_metadata={
            "model_type": training.model_type,
            "validation_scores_macro_f1": training.validation_scores,
            "validation_baselines_macro_f1": training.validation_baselines,
            **training.validation_details,
            "training_start": training.train_start,
            "training_end": training.train_end,
            "feature_count": len(FEATURE_COLUMNS),
            "macro_feature_columns": MACRO_FEATURE_COLUMNS,
            "horizon_days": horizon_days,
        },
        data_warnings=bundle.warnings,
    )

    if save_models:
        model_path = MODEL_DIR / f"{bundle.ticker.lower()}_{training.model_type.lower()}_{date.today().isoformat()}.joblib"
        joblib.dump(training.model, model_path)
        five_day_model_path = MODEL_DIR / f"{bundle.ticker.lower()}_five_day_forecast_{date.today().isoformat()}.joblib"
        joblib.dump(
            {
                "direction_model": five_day_training.direction_model,
                "return_model": five_day_training.return_model,
                "conditional_return_models": five_day_training.conditional_return_models,
            },
            five_day_model_path,
        )
    if save_outputs:
        save_signal_outputs(signal)
    return signal, training.report
