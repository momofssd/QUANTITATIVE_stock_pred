"""Reproducible collected-history evaluation: python -m src.machine_learning.evaluate."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..data_fetching import MarketDataBundle, load_market_data
from ..data_modeling import build_features, model_matrix, short_term_model_matrix
from ..utils import OUTPUT_DIR
from .five_day_forecast import generate_five_day_forecast


def evaluate_collected_history(bundle: MarketDataBundle, output_dir: Path) -> dict:
    """Compare feature sets on identical purged test dates; save metrics and predictions.

    This comparison does not select or tune models. Both runs use the new
    calibration/validation procedure, so differences isolate the feature set.
    """
    features = build_features(
        bundle.ticker_data, bundle.benchmark_primary_data, bundle.benchmark_secondary_data,
        bundle.vix_data, bundle.treasury_yield_data, bundle.oil_data,
    )
    matrices = {"original_features": model_matrix(features), "expanded_features": short_term_model_matrix(features)}
    masks = [matrix.notna().sum(axis=1) > max(3, int(matrix.shape[1] * 0.5)) for matrix in matrices.values()]
    common = masks[0] & masks[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ticker": bundle.ticker,
        "history_start": str(features.index[0].date()), "history_end": str(features.index[-1].date()),
        "horizon_trading_days": 5,
        "purpose": "feature_comparison_with_fixed_models_not_hyperparameter_selection",
        "data_warnings": bundle.warnings or [],
        "runs": {},
    }
    test_dates = None
    for name, matrix in matrices.items():
        matrix = matrix.copy()
        matrix.loc[~common] = np.nan
        forecast, training = generate_five_day_forecast(matrix, features, features.index[-1])
        if test_dates is not None and not training.validation_predictions.index.equals(test_dates):
            raise RuntimeError("Feature comparisons must share identical final test rows.")
        test_dates = training.validation_predictions.index
        ledger_path = output_dir / f"{bundle.ticker.lower()}_{name}_test_predictions.csv"
        training.validation_predictions.to_csv(ledger_path, index_label="entry_date")
        result["runs"][name] = {
            "feature_count": len(matrix.columns),
            "validation": training.validation,
            "purchase_evidence": forecast["purchase_evidence"],
            "test_predictions_file": ledger_path.name,
        }
        if name == "expanded_features":
            forecast_path = output_dir / f"{bundle.ticker.lower()}_forecast.json"
            forecast_path.write_text(json.dumps(forecast, indent=2, allow_nan=False), encoding="utf-8")
    report_path = output_dir / f"{bundle.ticker.lower()}_validation.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    """Load daily histories (cache first), evaluate each ticker, and save an aggregate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default=None, help="Exclusive history end; use a cached snapshot's date to reproduce it.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "short_term_validation")
    args = parser.parse_args()
    results = []
    for ticker in args.tickers:
        bundle = load_market_data(ticker, args.start_date, args.end_date)
        result = evaluate_collected_history(bundle, args.output_dir)
        results.append(result)
        metrics = result["runs"]["expanded_features"]["validation"]
        print(f"{ticker}: test accuracy {metrics['direction_accuracy']:.1%}, return MAE {metrics['return_mean_absolute_error']:.2%}, R2 {metrics['return_r2']}")
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
