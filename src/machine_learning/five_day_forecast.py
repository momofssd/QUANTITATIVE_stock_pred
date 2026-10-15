"""Five-session expected return, calibrated gain probability, and untouched-test validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..data_modeling.features import MACRO_FEATURE_COLUMNS
from ..data_modeling.labels import create_forward_return_target
from ..data_modeling.short_term_features import SHORT_TERM_FEATURE_COLUMNS
from ..utils import clean_float, confidence_label
from .validation import calibrate_direction_model, calibrated_error_radius, chronological_partitions, heldout_metrics
from .price_distribution import describe_distribution, distribution_validation


FIVE_DAY_HORIZON = 5
TREE_ESTIMATORS = 100
INTERVAL_COVERAGE = 0.8


@dataclass
class FiveDayTrainingResult:
    direction_model: Any
    return_model: Any
    conditional_return_models: dict[str, Any]
    direction_validation_macro_f1: float
    return_validation_mae: float
    conditional_return_validation_mae: dict[str, float]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    training_rows: int
    validation: dict = field(default_factory=dict)
    validation_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    interval_radius: float = 0.0
    feature_columns: list[str] = field(default_factory=list)
    live_fit: dict = field(default_factory=dict)
    calibration_residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    session_models: dict[int, Any] = field(default_factory=dict)


def _direction_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", ExtraTreesClassifier(
            n_estimators=TREE_ESTIMATORS, max_depth=8, min_samples_leaf=6,
            class_weight="balanced", random_state=42, n_jobs=1,
        )),
    ])


def _return_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", ExtraTreesRegressor(
            n_estimators=TREE_ESTIMATORS, max_depth=8, min_samples_leaf=6,
            random_state=42, n_jobs=1,
        )),
    ])


def train_five_day_models(
    X: pd.DataFrame,
    features: pd.DataFrame,
    horizon_days: int = FIVE_DAY_HORIZON,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.2,
) -> FiveDayTrainingResult:
    """Evaluate train/calibration/test chronologically, then fit a separate live pair.

    No test row trains an evaluation model, calibrates probabilities/ranges, or
    selects parameters. Purges prevent forward labels crossing both boundaries.
    Live models are refitted after evaluation with their own later calibration
    period; historical test scores belong to the pre-refit evaluation models.
    """
    if not isinstance(horizon_days, Integral) or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer.")
    if not X.index.equals(features.index) or not X.index.is_monotonic_increasing or not X.index.is_unique:
        raise ValueError("Features and X must share a unique, chronological index.")
    returns = create_forward_return_target(features, horizon_days=horizon_days)
    data = X.replace([np.inf, -np.inf], np.nan)
    valid = returns.notna() & np.isfinite(returns) & (data.notna().sum(axis=1) > max(3, int(data.shape[1] * 0.5)))
    data, returns = data.loc[valid].copy(), returns.loc[valid].astype(float)
    # A flat outcome is not a gain. This deliberately differs from the legacy
    # direction-label helper, which categorizes flat returns as bullish.
    labels = (returns > 0).astype(int)
    if len(data) < 180:
        raise ValueError("Not enough historical rows for short-term validation; need 180 usable labeled rows. Try an earlier start date.")
    train_idx, calibration_idx, test_idx = chronological_partitions(
        len(data), horizon_days, test_fraction, calibration_fraction,
    )
    X_train, X_cal, X_test = data.iloc[train_idx], data.iloc[calibration_idx], data.iloc[test_idx]
    y_train, y_cal, y_test = labels.iloc[train_idx], labels.iloc[calibration_idx], labels.iloc[test_idx]
    r_train, r_cal, r_test = returns.iloc[train_idx], returns.iloc[calibration_idx], returns.iloc[test_idx]

    base_direction = _direction_pipeline().fit(X_train, y_train)
    evaluation_direction = calibrate_direction_model(base_direction, X_cal, y_cal)
    evaluation_return = _return_pipeline().fit(X_train, r_train)
    calibration_prediction = np.maximum(evaluation_return.predict(X_cal), -0.99)
    radius = calibrated_error_radius(r_cal.to_numpy() - calibration_prediction, INTERVAL_COVERAGE)
    metrics, ledger = heldout_metrics(
        y_test, r_test, evaluation_direction.predict_proba(X_test)[:, 1],
        np.maximum(evaluation_return.predict(X_test), -0.99), r_train, radius, horizon_days,
    )
    target_dates = pd.Series(features.index, index=features.index).shift(-horizon_days)
    ledger["target_date"] = target_dates.loc[ledger.index]
    validation = {
        **metrics,
        "method": "purged_chronological_train_calibration_test",
        "purged_rows": horizon_days,
        "purge_boundaries": 2,
        "train_rows": len(X_train), "calibration_rows": len(X_cal),
        "train_start": str(X_train.index[0].date()), "train_end": str(X_train.index[-1].date()),
        "train_last_target_date": str(target_dates.loc[X_train.index[-1]].date()),
        "calibration_start": str(X_cal.index[0].date()), "calibration_end": str(X_cal.index[-1].date()),
        "calibration_last_target_date": str(target_dates.loc[X_cal.index[-1]].date()),
        "test_start": str(X_test.index[0].date()), "test_end": str(X_test.index[-1].date()),
        "test_last_target_date": str(target_dates.loc[X_test.index[-1]].date()),
        "calibration_status": evaluation_direction.calibration_status,
        "interval_nominal_coverage": INTERVAL_COVERAGE,
        "interval_calibration_radius_return": radius,
        "test_used_for_training_or_calibration": False,
        "metrics_model": "evaluation_models_before_live_refit",
        "target": "adjusted_close_return_strictly_above_zero_before_costs",
        "price_distribution": distribution_validation(
            ledger.predicted_return.to_numpy(), r_test.to_numpy(),
            r_cal.to_numpy() - calibration_prediction,
        ),
    }

    # Refit for today's forecast only after all evaluation predictions are frozen.
    live_cal_start = int(len(data) * (1 - calibration_fraction))
    live_train_end = live_cal_start - horizon_days
    live_base = _direction_pipeline().fit(data.iloc[:live_train_end], labels.iloc[:live_train_end])
    direction_model = calibrate_direction_model(live_base, data.iloc[live_cal_start:], labels.iloc[live_cal_start:])
    return_model = _return_pipeline().fit(data.iloc[:live_train_end], returns.iloc[:live_train_end])
    live_residuals = returns.iloc[live_cal_start:].to_numpy() - np.maximum(return_model.predict(data.iloc[live_cal_start:]), -0.99)
    live_radius = calibrated_error_radius(live_residuals, INTERVAL_COVERAGE)
    live_fit = {
        "policy": "separate_refit_with_later_calibration_after_evaluation",
        "uses_former_test_period": True,
        "train_rows": live_train_end, "calibration_rows": len(data) - live_cal_start,
        "train_end": str(data.index[live_train_end - 1].date()),
        "calibration_start": str(data.index[live_cal_start].date()),
        "calibration_end": str(data.index[-1].date()),
        "calibration_status": direction_model.calibration_status,
    }
    return FiveDayTrainingResult(
        direction_model=direction_model, return_model=return_model,
        conditional_return_models={}, conditional_return_validation_mae={},
        direction_validation_macro_f1=metrics["direction_macro_f1"],
        return_validation_mae=metrics["return_mean_absolute_error"],
        train_start=validation["train_start"], train_end=validation["train_end"],
        test_start=validation["test_start"], test_end=validation["test_end"], training_rows=len(data),
        validation=validation, validation_predictions=ledger, interval_radius=live_radius,
        feature_columns=list(data.columns), live_fit=live_fit,
        calibration_residuals=live_residuals,
    )


def _purchase_evidence(training: FiveDayTrainingResult) -> dict:
    """Expose fixed baseline checks; weak test results cannot endorse a current-price buy."""
    validation = training.validation
    baselines = validation["baselines"]
    purchase = validation["non_overlapping_purchase_check"]
    checks = {
        "enough_test_rows": validation["test_rows"] >= 100,
        "both_test_classes": validation["test_has_both_classes"],
        "probabilities_calibrated": validation["calibration_status"] == "sigmoid_calibrated" and training.live_fit["calibration_status"] == "sigmoid_calibrated",
        "beats_majority_accuracy": validation["direction_accuracy"] > baselines["training_majority_accuracy"],
        "positive_probability_skill": (validation["brier_skill_score"] or 0) > 0,
        "beats_return_baselines": validation["return_mean_absolute_error"] < min(baselines["zero_return_mae"], baselines["training_mean_return_mae"]),
        "enough_purchase_examples": purchase["qualifying_entries"] >= 10,
        "purchase_win_rate_above_base": purchase["win_rate"] is not None and purchase["win_rate"] > max(0.5, purchase["all_observations_gain_rate"]),
    }
    supported = all(checks.values())
    return {
        "supports_purchase": supported,
        "status": "BASELINE_EDGE" if supported else "INSUFFICIENT_VALIDATION_EDGE",
        "checks": checks,
        "note": "Historical baseline checks are a screening rule, not proof of a profitable strategy or future accuracy.",
    }


def generate_five_day_forecast(
    X: pd.DataFrame,
    features: pd.DataFrame,
    latest_index: pd.Timestamp,
    horizon_days: int = FIVE_DAY_HORIZON,
) -> tuple[dict, FiveDayTrainingResult]:
    """Forecast from the selected close using history available only through that row."""
    if latest_index not in features.index or latest_index not in X.index:
        raise ValueError("Forecast date must exist in both feature frames.")
    features, X = features.loc[:latest_index], X.loc[:latest_index]
    training = train_five_day_models(X, features, horizon_days=horizon_days)
    latest_X = X.loc[[latest_index]].replace([np.inf, -np.inf], np.nan)
    gain_probability = float(training.direction_model.predict_proba(latest_X)[0, 1])
    direction = "BULLISH" if gain_probability >= 0.5 else "BEARISH"
    confidence = max(gain_probability, 1 - gain_probability)
    current_price = float(features.loc[latest_index, "close"])
    projected_return = max(float(training.return_model.predict(latest_X)[0]), -0.99)
    possible_price = current_price * (1 + projected_return)
    price_direction = "BULLISH" if projected_return > 0 else "BEARISH"
    evidence = _purchase_evidence(training)
    warnings = []
    if training.direction_model.calibration_status != "sigmoid_calibrated":
        warnings.append("Too few examples of both outcomes for calibration; gain probability is a smoothed historical base rate.")
    if not evidence["supports_purchase"]:
        warnings.append("The final historical test does not pass all purchase-evidence checks; treat a bullish probability as an unconfirmed estimate.")
    if price_direction != direction:
        warnings.append("The probability model and expected-return regression disagree.")
    forecast = {
        "forecast_version": "4.0",
        "horizon_trading_days": horizon_days,
        "as_of_date": str(latest_index.date()),
        "predicted_direction": direction,
        "confidence_score": clean_float(confidence),
        "confidence_label": confidence_label(confidence) if not warnings else "LOW",
        "confidence_basis": training.direction_model.calibration_status,
        "probability_of_gain": clean_float(gain_probability),
        "probability_of_no_gain": clean_float(1 - gain_probability),
        "gain_definition": "Adjusted close strictly above the entry reference after the horizon, before fees, slippage, and taxes.",
        "entry_price_basis": "latest_available_daily_adjusted_close_not_live_quote",
        "direction_probabilities": {"bullish": clean_float(gain_probability), "bearish": clean_float(1 - gain_probability)},
        "current_price": clean_float(current_price, 2),
        "projected_return": clean_float(projected_return),
        "expected_gain_per_share": clean_float(possible_price - current_price, 2),
        "possible_price_in_5_trading_days": clean_float(possible_price, 2),
        "price_estimate_basis": "unconditional_forward_return_regression",
        "unconditional_expected_return": clean_float(projected_return),
        "unconditional_expected_price_in_5_trading_days": clean_float(possible_price, 2),
        "price_projection_direction": price_direction,
        "direction_price_models_agree": direction == price_direction,
        "possible_price_range": {
            "low": clean_float(max(current_price * (1 + projected_return - training.interval_radius), 0), 2),
            "high": clean_float(current_price * (1 + projected_return + training.interval_radius), 2),
            "basis": "chronological_calibration_absolute_error_quantile",
            "nominal_coverage": INTERVAL_COVERAGE,
            "heldout_observed_coverage": clean_float(training.validation["interval_observed_coverage"]),
        },
        "validation": training.validation,
        "purchase_evidence": evidence,
        "warnings": warnings,
        "model_metadata": {
            "direction_model_type": "ExtraTreesClassifierWithSigmoidCalibration",
            "price_model_type": "ExtraTreesRegressor",
            "price_model_conditioning": "all_outcomes",
            "training_rows": training.training_rows,
            "feature_count": len(X.columns),
            "feature_columns": training.feature_columns,
            "short_term_feature_columns": [name for name in SHORT_TERM_FEATURE_COLUMNS if name in X.columns],
            "macro_feature_columns": [name for name in MACRO_FEATURE_COLUMNS if name in X.columns],
            "live_fit": training.live_fit,
        },
        "interpretation_note": (
            "Gain probability estimates a positive return from the latest daily close before costs. "
            "The possible price is an unconditional regression estimate. The 80% nominal range is calibrated on earlier errors; "
            "see final-test coverage and accuracy separately. Serial dependence and market changes can reduce future coverage."
        ),
    }
    sessions = []
    for session in range(1, horizon_days + 1):
        fitted = training if session == horizon_days else train_five_day_models(X, features, horizon_days=session)
        prediction = max(float(fitted.return_model.predict(latest_X)[0]), -0.99)
        probability = float(fitted.direction_model.predict_proba(latest_X)[0, 1])
        sessions.append({
            "session": session,
            "expected_close": clean_float(current_price * (1 + prediction), 2),
            "probability_of_gain": probability,
            "predicted_direction": "BULLISH" if probability >= 0.5 else "BEARISH",
            "confidence_basis": fitted.direction_model.calibration_status,
            "price_distribution": describe_distribution(current_price, prediction, fitted.calibration_residuals),
            "calibrated_price_range": {
                "low": clean_float(max(current_price * (1 + prediction - fitted.interval_radius), 0), 2),
                "high": clean_float(current_price * (1 + prediction + fitted.interval_radius), 2),
                "nominal_coverage": INTERVAL_COVERAGE,
                "heldout_observed_coverage": fitted.validation["interval_observed_coverage"],
            },
            "validation": fitted.validation,
        })
        # Do not insert the parent result into itself (serialized artifacts).
        if session != horizon_days:
            training.session_models[session] = fitted
    forecast["session_forecasts"] = sessions
    forecast["price_distribution"] = sessions[-1]["price_distribution"]
    forecast["session_forecast_note"] = "Each session is modeled separately from the same as-of close, not from the prior predicted session. Session numbers refer to future exchange sessions, not calendar dates."
    return forecast, training
