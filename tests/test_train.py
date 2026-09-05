from src.train import load_dataset, build_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def test_dataset_loads_and_labels_are_binary():
    df = load_dataset("data/sms.tsv")
    assert len(df) > 5000
    assert set(df["label"].unique()) == {0, 1}


def test_pipeline_reaches_minimum_quality():
    df = load_dataset("data/sms.tsv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    # Sanity threshold well below the ~0.976 actually observed, so this
    # only fails if something breaks the pipeline, not on normal variance.
    assert accuracy > 0.9
