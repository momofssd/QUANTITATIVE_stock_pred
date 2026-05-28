import numpy as np
import pandas as pd

from src.indicators import atr, rsi


def test_indicator_calculation_outputs_expected_lengths():
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.Series(np.linspace(100, 120, 40), index=index)
    df = pd.DataFrame(
        {
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
        }
    )

    assert len(rsi(close)) == 40
    assert len(atr(df)) == 40
    assert rsi(close).dropna().between(0, 100).all()
    assert (atr(df).dropna() > 0).all()
