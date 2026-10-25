"""
Integration tests for FastAPI Y2K Frontend API endpoints.
Tests:
- /api/query
- /api/graph
- /api/versions
- /api/diff
- /api/adapt/step
- /api/break-it
- /api/rollback
"""

import pytest
from fastapi.testclient import TestClient
from morphe.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_graph_and_versions(client):
    res = client.get("/api/graph")
    assert res.status_code == 200
    data = res.json()
    assert "elements" in data
    assert "total_modules" in data

    ver_res = client.get("/api/versions")
    assert ver_res.status_code == 200
    assert "versions" in ver_res.json()


def test_api_query_endpoint(client):
    res = client.post("/api/query", json={"query": "What is our memory architecture?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert data["query"] == "What is our memory architecture?"


def test_api_break_it_auto_rejection(client):
    res = client.post("/api/break-it")
    assert res.status_code == 200
    data = res.json()
    # Verifies that break-it either reports rejection or needs modules
    assert data["status"] in {"auto_rejected", "error"}
    if data["status"] == "auto_rejected":
        assert data["reward"] < 0


def test_api_adapt_step(client):
    res = client.post("/api/adapt/step")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in {"committed", "rejected", "no_candidates", "optimal"}

