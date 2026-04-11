"""API tests for FastAPI endpoints using a lightweight mocked system."""
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.main as main_module


class FakeSystem:
    def analyze_query(self, user_query: str):
        return {
            "status": "success",
            "analysis": {
                "problem": f"Analyzed: {user_query}",
                "root_causes": ["cause_a"],
                "suggestions": ["suggestion_a"],
                "confidence": "high",
                "expected_improvement": "50%",
            },
            "raw_response": "{}",
        }

    def detect_anomaly(self, user_query: str):
        return {
            "status": "success",
            "is_anomaly": True,
            "confidence": "medium",
            "reason": f"Detected anomaly for query: {user_query}",
            "matched_case": "Anomaly: Latency Spike.",
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "PerformanceAnalysisSystem", FakeSystem)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_query_endpoint_success(client):
    response = client.post("/analyze/query", json={"query": "Why is this query slow?"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "success"
    assert "analysis" in payload
    assert payload["analysis"]["confidence"] == "high"


def test_detect_anomaly_endpoint_success(client):
    response = client.post("/detect/anomaly", json={"query": "Latency spike observed"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["is_anomaly"] is True
    assert payload["confidence"] == "medium"


def test_analyze_query_endpoint_validation(client):
    response = client.post("/analyze/query", json={"query": "   "})
    assert response.status_code == 400
    assert response.json() == {"detail": "query must not be empty"}
