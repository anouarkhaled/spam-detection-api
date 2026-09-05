import time
from collections import deque
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(
    title="Spam Detection API",
    description="TF-IDF + Logistic Regression spam classifier, served with monitoring.",
)

# In-memory monitoring store. Resets on restart -- fine for a portfolio demo,
# would be Prometheus/a real time-series DB in an actual production service.
MAX_HISTORY = 500
_history: deque = deque(maxlen=MAX_HISTORY)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    spam_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        result = predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _history.append(
        {
            "ts": time.time(),
            "label": result["label"],
            "spam_probability": result["spam_probability"],
            "text_length": len(req.text),
        }
    )
    return result


@app.get("/stats")
def stats():
    items = list(_history)
    total = len(items)
    spam_count = sum(1 for i in items if i["label"] == "spam")
    avg_confidence = (
        round(sum(i["spam_probability"] for i in items) / total, 4) if total else None
    )
    return {
        "total_predictions": total,
        "spam_count": spam_count,
        "ham_count": total - spam_count,
        "avg_spam_probability": avg_confidence,
        "recent": list(items)[-50:],
    }


_DASHBOARD_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Spam API - Monitoring</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; background: #0f151c; color: #e8edf2; margin: 0; padding: 24px; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  .sub { color: #94a3b0; font-size: 0.85rem; margin-bottom: 24px; }
  .cards { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .card { background: #161f28; border: 1px solid #2a3641; border-radius: 8px; padding: 14px 18px; min-width: 140px; }
  .card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; color: #94a3b0; }
  .card .value { font-size: 1.6rem; font-weight: 600; margin-top: 4px; }
  #chart-wrap { max-width: 700px; }
  form { margin-top: 32px; display: flex; gap: 8px; max-width: 700px; }
  input { flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid #2a3641; background: #161f28; color: #e8edf2; }
  button { padding: 8px 16px; border-radius: 6px; border: none; background: #e2a463; color: #16212e; font-weight: 600; cursor: pointer; }
  #result { margin-top: 12px; font-size: 0.9rem; }
</style>
</head>
<body>
  <h1>Spam Detection API - Monitoring</h1>
  <div class="sub">Statistiques en memoire depuis le dernier redemarrage du serveur.</div>

  <div class="cards">
    <div class="card"><div class="label">Predictions</div><div class="value" id="total">-</div></div>
    <div class="card"><div class="label">Spam</div><div class="value" id="spam">-</div></div>
    <div class="card"><div class="label">Ham</div><div class="value" id="ham">-</div></div>
    <div class="card"><div class="label">Confiance moy.</div><div class="value" id="conf">-</div></div>
  </div>

  <div id="chart-wrap"><canvas id="chart" height="120"></canvas></div>

  <form id="predict-form">
    <input id="text-input" type="text" placeholder="Coller un message a tester..." />
    <button type="submit">Predire</button>
  </form>
  <div id="result"></div>

<script>
let chart;
async function refresh() {
  const r = await fetch('/stats');
  const s = await r.json();
  document.getElementById('total').textContent = s.total_predictions;
  document.getElementById('spam').textContent = s.spam_count;
  document.getElementById('ham').textContent = s.ham_count;
  document.getElementById('conf').textContent = s.avg_spam_probability ?? '-';

  const labels = s.recent.map((_, i) => i + 1);
  const data = s.recent.map(x => x.spam_probability);
  if (!chart) {
    const ctx = document.getElementById('chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [{ label: 'Probabilite spam (50 dernieres)', data, borderColor: '#e2a463', backgroundColor: 'rgba(226,164,99,0.15)', fill: true, tension: 0.25, pointRadius: 2 }] },
      options: { scales: { y: { min: 0, max: 1, ticks: { color: '#94a3b0' } }, x: { ticks: { color: '#94a3b0' } } }, plugins: { legend: { labels: { color: '#e8edf2' } } } }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }
}
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = document.getElementById('text-input').value;
  if (!text) return;
  const r = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
  const data = await r.json();
  document.getElementById('result').textContent = r.ok ? `-> ${data.label} (probabilite spam: ${data.spam_probability})` : `Erreur: ${data.detail}`;
  refresh();
});
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML
