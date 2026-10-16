"""API tests with a throw-away SQLite database (no models are loaded)."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from csi import service
from csi.api import security
from csi.api.main import app
from csi.config import Paths, settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    paths = Paths(artifacts=tmp_path, warehouse=tmp_path / "test.db")
    monkeypatch.setattr(service, "settings", dataclasses.replace(settings, paths=paths))
    monkeypatch.setattr(security, "limiter", security.RateLimiter(60))
    return TestClient(app)


def test_feedback_endpoint_saves_rating(client):
    triage_id = service.log_rows([{"queue": "IT Support", "groundedness": 0.8}])[0]
    r = client.post("/v1/feedback", json={"triage_id": triage_id, "helpful": True})
    assert r.status_code == 201 and r.json()["feedback_id"] == 1
    assert client.get("/v1/feedback/summary").json()["helpful_rate"] == 1.0


def test_feedback_endpoint_rejects_unknown_ticket_and_queue(client):
    assert client.post("/v1/feedback", json={"triage_id": 42, "helpful": True}).status_code == 404
    r = client.post("/v1/feedback", json={"triage_id": 1, "helpful": False, "corrected_queue": "Sales"})
    assert r.status_code == 422


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(security, "settings", dataclasses.replace(settings, api_key="s3cret"))
    assert client.get("/v1/feedback/summary").status_code == 401
    assert client.get("/v1/feedback/summary", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/feedback/summary", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter(2))
    codes = [client.get("/v1/feedback/summary").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert int(client.get("/v1/feedback/summary").headers["Retry-After"]) >= 1


def test_rate_limiter_window_slides():
    rl = security.RateLimiter(limit=2, window_s=60)
    assert rl.retry_after("a", now=0) == 0 and rl.retry_after("a", now=1) == 0
    assert rl.retry_after("a", now=2) == 58.0          # third call inside the window is blocked
    assert rl.retry_after("b", now=2) == 0             # other clients are not affected
    assert rl.retry_after("a", now=60.5) == 0          # first hit has left the window
