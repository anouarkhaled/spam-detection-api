import os

import joblib

_MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"No model found at {_MODEL_PATH}. Run `python src/train.py` first."
            )
        _pipeline = joblib.load(_MODEL_PATH)
    return _pipeline


def predict(text: str) -> dict:
    pipeline = get_pipeline()
    proba = pipeline.predict_proba([text])[0]
    spam_probability = float(proba[1])
    return {
        "label": "spam" if spam_probability >= 0.5 else "ham",
        "spam_probability": round(spam_probability, 4),
    }
