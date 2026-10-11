"""Bindu prediction endpoint (Vercel serverless function).

Model: Random Forest (300 trees), Grid Search, ETC feature set, trained in Notebook 04 of the study
(models/Optimised/etc/rf_grid.pkl) and exported to models/bindu_rf_etc.npz by tools/export_model.py.
The forest is evaluated with numpy only. The input is the raw answer to each question; the StandardScaler is
applied first and the value is cast to float32 before each split test, exactly as scikit-learn does, so
the output matches the trained pipeline.
"""
import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bindu_rf_etc.npz")

# Order is fixed by the trained model: PSS-10 items 2 and 10, GAD-7 items 1-7, PHQ-9 items 2-7.
FEATURE_ORDER = ["PSS2", "PSS10",
                 "GAD1", "GAD2", "GAD3", "GAD4", "GAD5", "GAD6", "GAD7",
                 "PHQ2", "PHQ3", "PHQ4", "PHQ5", "PHQ6", "PHQ7"]
MAX_ANSWER    = {f: (4 if f.startswith("PSS") else 3) for f in FEATURE_ORDER}
CLASS_NAMES   = ["Stable", "Challenged", "Critical"]

_model = None


def get_model():
    global _model
    if _model is None:
        z = np.load(MODEL_PATH)
        _model = {k: z[k] for k in z.files}
        _model["depth"] = 64   # upper bound on tree depth; leaves point to themselves, so extra steps are harmless
    return _model


def predict_proba(x):
    """x: sequence of 15 raw answers in FEATURE_ORDER -> array of 3 class probabilities."""
    m = get_model()
    x = ((np.asarray(x, dtype=np.float64) - m["mean"]) / m["scale"]).astype(np.float32)
    node = m["roots"].astype(np.int64)
    for _ in range(m["depth"]):
        f = m["feature"][node]
        go_left = x[f] <= m["threshold"][node]
        nxt = np.where(go_left, m["left"][node], m["right"][node])
        if np.array_equal(nxt, node):
            break
        node = nxt
    return m["value"][node].astype(np.float64).mean(axis=0)


def validate(responses):
    missing = [f for f in FEATURE_ORDER if f not in responses]
    if missing:
        raise ValueError("Missing answers: " + ", ".join(missing))
    values = []
    for f in FEATURE_ORDER:
        v = responses[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v) or not 0 <= int(v) <= MAX_ANSWER[f]:
            raise ValueError(f"Invalid answer for {f}")
        values.append(int(v))
    return values


def _cors(h):
    h.send_header("Access-Control-Allow-Origin",  "*")
    h.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def _respond(h, status, payload):
    body = json.dumps(payload).encode()
    h.send_response(status)
    _cors(h)
    h.send_header("Content-Type",   "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        _cors(self)
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            values = validate(data.get("responses", {}))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return _respond(self, 400, {"error": str(e)})
        try:
            probs = predict_proba(values)
            pred_idx = int(np.argmax(probs))
            _respond(self, 200, {
                "prediction":       CLASS_NAMES[pred_idx],
                "prediction_index": pred_idx,
                "confidence":       round(float(probs[pred_idx]) * 100, 1),
                "probabilities":    {CLASS_NAMES[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)},
            })
        except Exception:
            _respond(self, 500, {"error": "Prediction failed"})

    def log_message(self, *args):
        pass