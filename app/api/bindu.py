import json
import os
import warnings
import numpy as np
import joblib
from http.server import BaseHTTPRequestHandler

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "gb_randomized.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

_model  = None
_scaler = None

def get_artifacts():
    global _model, _scaler
    if _model is None:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler

FEATURE_ORDER = [
    "PSS3",
    "GAD1", "GAD2", "GAD3", "GAD4", "GAD5", "GAD6", "GAD7",
    "PHQ2", "PHQ3", "PHQ4", "PHQ5", "PHQ6", "PHQ7", "PHQ8",
]

CLASS_NAMES = ["Stable", "Challenged", "Critical"]


# ── Module-level helpers (work regardless of handler subclass) ─────────────────
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
            length    = int(self.headers.get("Content-Length", 0))
            body      = self.rfile.read(length)
            data      = json.loads(body)
            responses = data.get("responses", {})

            features = np.array(
                [[float(responses.get(f, 0)) for f in FEATURE_ORDER]]
            )

            model, scaler = get_artifacts()
            scaled = scaler.transform(features)

            pred_idx   = int(model.predict(scaled)[0])
            probs      = model.predict_proba(scaled)[0].tolist()
            confidence = round(probs[pred_idx] * 100, 1)

            _respond(self, 200, {
                "prediction":       CLASS_NAMES[pred_idx],
                "prediction_index": pred_idx,
                "confidence":       confidence,
                "probabilities": {
                    CLASS_NAMES[i]: round(p * 100, 1)
                    for i, p in enumerate(probs)
                },
            })

        except Exception as e:
            _respond(self, 500, {"error": str(e)})

    def log_message(self, *args):
        pass