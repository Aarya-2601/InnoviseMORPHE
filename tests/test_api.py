"""
Unit tests for MORPHE FastAPI endpoints:
- GET / (HTML Dashboard)
- POST /api/ingest
- POST /api/retrieve
- POST /api/adapt
- POST /api/rollback
- GET /api/topology
- GET /api/history
- GET /api/policy_stats
- POST /api/chaos/break_it
- POST /api/chaos/drift_it
- POST /api/audit
"""

from fastapi.testclient import TestClient
from morphe.api.app import app

client = TestClient(app)


def test_index_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "MORPHE" in response.text
    assert "cytoscape" in response.text


def test_api_topology_and_history():
    res_top = client.get("/api/topology")
    assert res_top.status_code == 200
    data_top = res_top.json()
    assert "elements" in data_top
    assert len(data_top["elements"]) > 0

    res_hist = client.get("/api/history")
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert len(data_hist) >= 1
    assert data_hist[0]["version_id"] == "v1"


def test_api_retrieve():
    res = client.post("/api/retrieve", json={"query": "OAuth2 PKCE authorization", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "trace" in data
    assert len(data["trace"]["modules_activated"]) >= 1


def test_api_ingest():
    res = client.post("/api/ingest", json={
        "text": "New dynamic security rule for rate-limiting on microservices.",
        "project": "ProjectAlpha_Auth",
        "memory_type": "tech_spec",
        "tags": ["rate_limit", "security"],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "record_id" in data


def test_api_policy_stats():
    res = client.get("/api/policy_stats")
    assert res.status_code == 200
    data = res.json()
    assert "cumulative_regret" in data
    assert "arms" in data


def test_api_chaos_endpoints():
    res_drift = client.post("/api/chaos/drift_it")
    assert res_drift.status_code == 200

    res_break = client.post("/api/chaos/break_it")
    assert res_break.status_code == 200
    assert "protection_result" in res_break.json()


def test_api_audit():
    res = client.post("/api/audit")
    assert res.status_code == 200
    data = res.json()
    assert "pass_rate" in data
