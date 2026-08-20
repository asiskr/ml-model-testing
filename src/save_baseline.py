import json

from src.clean_data import clean_data
from src.config import (
    BASELINE_JSON,
    MAX_DEPTH,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_CSV,
)
from src.load_data import load_data, separate_features_target
from src.train_model import evaluate_model, split_data, train_random_forest


def build_baseline():
    df = load_data()
    X, y = separate_features_target(df)
    X = clean_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, max_depth=MAX_DEPTH)
    _, accuracy, _ = evaluate_model(model, X_test, y_test)

    return {
        "model": type(model).__name__,
        "accuracy": accuracy,
        "max_depth": MAX_DEPTH,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dataset": TRAIN_CSV.name,
    }


def save_baseline():
    baseline = build_baseline()
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


if __name__ == "__main__":
    print(json.dumps(save_baseline(), indent=2))
