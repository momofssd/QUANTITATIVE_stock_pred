"""Empirical closing-price distributions from chronological calibration errors.

These are marginal distributions for each horizon, not simulated joint paths or
intraday touch probabilities. Fixed bins are chosen before evaluation.
"""

import numpy as np

from ..utils import clean_float


RETURN_EDGES = np.array([-0.03, -0.01, 0.01, 0.03])
BIN_LABELS = ["STRONGLY_BEARISH", "BEARISH", "SIDEWAYS", "BULLISH", "STRONGLY_BULLISH"]


def return_samples(prediction, residuals):
    errors = np.asarray(residuals, dtype=float)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Price distributions require finite calibration residuals.")
    return np.maximum(np.asarray(prediction)[..., None] + errors, -1.0)


def bin_probabilities(samples):
    # Each interior boundary belongs to the bin on its right.
    categories = np.searchsorted(RETURN_EDGES, samples, side="right")
    return np.stack([(categories == i).mean(axis=-1) for i in range(5)], axis=-1)


def distribution_validation(prediction, actual, residuals):
    probabilities = bin_probabilities(return_samples(prediction, residuals))
    outcomes = np.eye(5)[np.searchsorted(RETURN_EDGES, actual, side="right")]
    return {
        "method": "fixed_return_bins_with_pretest_calibration_residuals",
        "multiclass_brier_score": float(np.square(probabilities - outcomes).sum(axis=1).mean()),
        "mean_predicted_bin_probabilities": probabilities.mean(axis=0).tolist(),
        "observed_bin_frequencies": outcomes.mean(axis=0).tolist(),
        "bin_order": BIN_LABELS,
        "test_used_for_distribution_calibration": False,
    }


def describe_distribution(price, prediction, residuals):
    samples = return_samples(prediction, residuals)
    probabilities = bin_probabilities(samples)
    edges = [-1.0, *RETURN_EDGES, None]
    bins = [{
        "label": label,
        "low_return_inclusive": edges[i],
        "high_return_exclusive": edges[i + 1],
        "low_price": clean_float(price * (1 + edges[i]), 2),
        "high_price": clean_float(price * (1 + edges[i + 1]), 2) if edges[i + 1] is not None else None,
        "probability": float(probabilities[i]),
    } for i, label in enumerate(BIN_LABELS)]
    bearish, sideways, bullish = float(probabilities[:2].sum()), float(probabilities[2]), float(probabilities[3:].sum())
    bias = "BULLISH" if bullish > max(bearish, sideways) else "BEARISH" if bearish > max(bullish, sideways) else "SIDEWAYS_OR_MIXED"
    return {
        "method": "return_regression_plus_empirical_calibration_residuals",
        "calibration_samples": int(len(samples)),
        "bins": bins,
        "bias": bias,
        "scenario_probabilities": {"bearish": bearish, "sideways": sideways, "bullish": bullish},
        "probability_of_gain": float((samples > 0).mean()),
        "price_quantiles": {f"p{q}": clean_float(price * (1 + np.quantile(samples, q / 100)), 2) for q in (10, 25, 50, 75, 90)},
        "note": "Closing-price estimates relative to today's adjusted close; sideways means -1% to +1%. Bin probabilities are empirical estimates, not separately calibrated classifier probabilities. No intraday touch or bounce probability is modeled. Overlapping historical returns and changing volatility limit reliability.",
    }
