"""API tests with a throw-away SQLite database (no models are loaded)."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from csi import service
from csi.api.main import app
from csi.config import Paths, settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    paths = Paths(artifacts=tmp_path, warehouse=tmp_path / "test.db")
    monkeypatch.setattr(service, "settings", dataclasses.replace(settings, paths=paths))
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
