from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_obvious_spam():
    r = client.post(
        "/predict",
        json={"text": "WINNER!! You have won a FREE prize, call 09061701461 to claim now!"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "spam"
    assert 0 <= body["spam_probability"] <= 1


def test_predict_obvious_ham():
    r = client.post("/predict", json={"text": "Are we still on for lunch tomorrow?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "ham"


def test_predict_rejects_empty_text():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_stats_after_predictions():
    client.post("/predict", json={"text": "Call now to win a free prize"})
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1
    assert "recent" in body


def test_dashboard_serves_html():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
