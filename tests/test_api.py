"""FastAPI endpoint tests using the TestClient (synchronous, no server)."""

import os

os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    """Behavioral test: health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "modelLoaded" in body
    assert "modelVersion" in body
    assert "mockInference" in body
    assert body["mockInference"] is True


def test_predict_returns_expected_shape():
    """Behavioral test: predict returns expected shape."""
    resp = client.post(
        "/predict",
        json={"topic": "economics", "text": "I disagree with this approach."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictedLabel"] in {
        "supportive",
        "opposed",
        "neutral",
        "mixed",
        "unclear",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["labelScores"].keys()) == {
        "supportive",
        "opposed",
        "neutral",
        "mixed",
        "unclear",
    }


def test_predict_validates_input():
    """A blank field must fail validation as exactly 400 INVALID_INPUT.

    The integration contract collapses pydantic schema failures to 400
    (there is no 422 on this service), so we assert the EXACT code and
    envelope rather than the old ``in (400, 422)`` which masked drift
    between the handler and the contract (audit D4 / §9).
    """
    resp = client.post("/predict", json={"topic": "", "text": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "INVALID_INPUT"


def test_predict_missing_field():
    """A missing required field must also be exactly 400 INVALID_INPUT."""
    resp = client.post("/predict", json={"topic": "ai"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "INVALID_INPUT"
