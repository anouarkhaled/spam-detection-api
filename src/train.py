"""Train a spam-classification baseline and save the model + metrics.

Usage: python src/train.py --data data/sms.tsv --out model
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", classifier),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, target_names=["ham", "spam"], output_dict=True),
    }


def main(data_path: str, out_dir: str):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    candidates = {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "naive_bayes": build_pipeline(MultinomialNB()),
    }

    os.makedirs(out_dir, exist_ok=True)
    results = {}
    best_name, best_pipeline, best_f1 = None, None, -1.0

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, metrics["f1"]

    joblib.dump(best_pipeline, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nBest model: {best_name} (f1={best_f1:.4f}) saved to {out_dir}/model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sms.tsv")
    parser.add_argument("--out", default="model")
    args = parser.parse_args()
    main(args.data, args.out)
