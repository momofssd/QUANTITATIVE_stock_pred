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
