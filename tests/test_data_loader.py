import os
import sys
import time
from types import SimpleNamespace

import pandas as pd

from src import data_loader


def sample_frame():
    index = pd.to_datetime(["2026-05-22", "2026-05-26"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=index,
    )


def test_download_ohlcv_uses_direct_chart_source(monkeypatch, tmp_path):
    chart_frame = sample_frame()
    chart_calls = []

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    def chart_download(ticker, start, end):
        chart_calls.append((ticker, start, end))
        return chart_frame

    monkeypatch.setattr(data_loader, "_download_yahoo_chart", chart_download)

    result = data_loader.download_ohlcv("SMH", "2026-05-01", cache=False)

    assert result.equals(chart_frame)
    assert chart_calls == [("SMH", "2026-05-01", None)]


def test_download_ohlcv_uses_yfinance_when_chart_source_fails(monkeypatch, tmp_path):
    fallback_frame = sample_frame()
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        data_loader,
        "_download_yahoo_chart",
        lambda *_args: (_ for _ in ()).throw(data_loader.MarketDataUnavailableError("chart failed")),
    )
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda *_args, **_kwargs: fallback_frame),
    )

    result = data_loader.download_ohlcv("SMH", "2026-05-01", cache=False)

    assert result.equals(fallback_frame)


def _counting_chart(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_loader, "_download_yahoo_chart", lambda *args: calls.append(args) or sample_frame().rename_axis("Date"))
    return calls


def test_open_ended_cache_expires_so_the_latest_session_is_refetched(monkeypatch, tmp_path):
    calls = _counting_chart(monkeypatch, tmp_path)
    data_loader.download_ohlcv("SMH", "2026-05-01")
    data_loader.download_ohlcv("SMH", "2026-05-01")
    assert len(calls) == 1
    cached = next((tmp_path / "data" / "cache").glob("SMH_*_latest.csv"))
    stale = time.time() - data_loader.LIVE_CACHE_SECONDS - 1
    os.utime(cached, (stale, stale))
    data_loader.download_ohlcv("SMH", "2026-05-01")
    assert len(calls) == 2


def test_past_end_date_cache_is_reused_regardless_of_age(monkeypatch, tmp_path):
    calls = _counting_chart(monkeypatch, tmp_path)
    data_loader.download_ohlcv("SMH", "2026-05-01", "2026-05-27")
    cached = next((tmp_path / "data" / "cache").glob("SMH_*_2026-05-27.csv"))
    os.utime(cached, (0, 0))
    data_loader.download_ohlcv("SMH", "2026-05-01", "2026-05-27")
    assert len(calls) == 1


def test_unsettled_latest_session_is_filled_from_the_quote():
    frame = sample_frame()
    frame.iloc[-1, frame.columns.get_loc("Close")] = None
    frame.iloc[-1, frame.columns.get_loc("Adj Close")] = None
    meta = {
        "regularMarketPrice": 104.5,
        "regularMarketDayHigh": 105.0,
        "regularMarketDayLow": 100.5,
        "regularMarketVolume": 1500,
        "regularMarketTime": int(pd.Timestamp("2026-05-26 20:00", tz="UTC").timestamp()),
    }
    filled = data_loader._fill_unsettled_session(frame, meta)
    assert filled["Close"].iloc[-1] == 104.5 and filled["Adj Close"].iloc[-1] == 104.5
    # Values Yahoo did send are kept.
    assert filled["High"].iloc[-1] == 103.0


def test_quote_from_another_day_does_not_fill_the_latest_session():
    frame = sample_frame()
    frame.iloc[-1, frame.columns.get_loc("Close")] = None
    meta = {"regularMarketPrice": 104.5, "regularMarketTime": int(pd.Timestamp("2026-05-27 20:00", tz="UTC").timestamp())}
    assert pd.isna(data_loader._fill_unsettled_session(frame, meta)["Close"].iloc[-1])
