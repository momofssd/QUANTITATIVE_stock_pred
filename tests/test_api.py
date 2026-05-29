from fastapi.testclient import TestClient

from src.api import PredictRequest, app
from src.data_loader import MarketDataUnavailableError


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_api_health_alias():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_request_defaults_to_interactive_extra_trees_model():
    assert PredictRequest(ticker="SMH").model == "extratrees"


def test_predict_reports_market_data_provider_failure(monkeypatch):
    def unavailable_signal(**_kwargs):
        raise MarketDataUnavailableError("Unable to download OHLCV data for SMH.")

    monkeypatch.setattr("src.api.generate_quant_signal", unavailable_signal)
    client = TestClient(app)
    response = client.post("/api/predict", json={"ticker": "SMH"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to download OHLCV data for SMH."


def test_predict_rejects_parallel_analysis_request(monkeypatch):
    class BusyLock:
        def acquire(self, blocking=False):
            return False

    monkeypatch.setattr("src.api.ANALYSIS_LOCK", BusyLock())
    client = TestClient(app)
    response = client.post("/api/predict", json={"ticker": "SMH"})

    assert response.status_code == 429
    assert response.json()["detail"] == "Another analysis is running. Wait for it to finish and try again."


def test_interpret_reports_configuration_failure(monkeypatch):
    def unavailable_interpreter(_signal_json):
        raise RuntimeError("Missing Gemini API key.")

    monkeypatch.setattr("src.api.interpret_signal_with_gemini", unavailable_interpreter)
    client = TestClient(app)
    response = client.post("/api/interpret", json={"ticker": "SMH"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Missing Gemini API key."
