def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_routes_lists_all_endpoints(client):
    response = client.get("/api/routes")
    assert response.status_code == 200
    paths = [r[1] for r in response.json()["routes"]]
    assert "/api/health" in paths
    assert "/api/chat/{user_id}" in paths
    assert "/api/products/search" in paths
