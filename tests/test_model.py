import json

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.config import (
    BASELINE_JSON,
    MAX_DEPTH,
    MAX_OVERFIT_GAP,
    MIN_ACCURACY,
    REGRESSION_TOLERANCE,
)
from src.train_model import evaluate_model, train_decision_tree

pytestmark = pytest.mark.model

GOLDEN_MAX_DEPTH = 5
GOLDEN_PREDICTIONS = [0, 0, 0, 1, 1]


def test_model_accuracy_above_threshold(model_data, trained_model):
    _, X_test, _, y_test = model_data

    _, accuracy, _ = evaluate_model(trained_model, X_test, y_test)

    assert accuracy >= MIN_ACCURACY, (
        f"accuracy {accuracy:.2%} is below the {MIN_ACCURACY:.2%} gate"
    )


def test_model_not_overfitting(model_data, trained_model):
    X_train, X_test, y_train, y_test = model_data

    _, train_accuracy, _ = evaluate_model(trained_model, X_train, y_train)
    _, test_accuracy, _ = evaluate_model(trained_model, X_test, y_test)
    gap = train_accuracy - test_accuracy

    assert gap < MAX_OVERFIT_GAP, (
        f"train {train_accuracy:.2%} vs test {test_accuracy:.2%} "
        f"is a {gap:.2%} gap, over the {MAX_OVERFIT_GAP:.2%} limit"
    )


def test_model_deterministic(model_data, trained_model):
    _, X_test, _, _ = model_data

    first = trained_model.predict(X_test)
    second = trained_model.predict(X_test)

    assert (first == second).all(), "same input produced different predictions"


def test_single_row_prediction(model_data, trained_model):
    _, X_test, _, _ = model_data

    predictions = trained_model.predict(X_test.head(1))

    assert len(predictions) == 1


def test_prediction_shape(model_data, trained_model):
    _, X_test, _, _ = model_data

    predictions = trained_model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1}), (
        f"model returned labels outside 0/1: {set(predictions)}"
    )


def test_model_rejects_wrong_columns(model_data, trained_model):
    _, X_test, _, _ = model_data

    missing_a_feature = X_test.drop(columns=["Age"])

    with pytest.raises(ValueError):
        trained_model.predict(missing_a_feature)


def test_random_forest_beats_decision_tree(model_data, trained_model):
    X_train, X_test, y_train, y_test = model_data

    decision_tree = train_decision_tree(X_train, y_train, max_depth=MAX_DEPTH)

    accuracy_rf = accuracy_score(y_test, trained_model.predict(X_test))
    accuracy_dt = accuracy_score(y_test, decision_tree.predict(X_test))

    assert accuracy_rf >= accuracy_dt, (
        f"random forest {accuracy_rf:.2%} lost to decision tree {accuracy_dt:.2%}"
    )


def test_duplicate_row_same_prediction(model_data, trained_model):
    _, X_test, _, _ = model_data

    one_passenger = X_test.head(1)
    duplicated = pd.concat([one_passenger] * 3)

    predictions = trained_model.predict(duplicated)

    assert len(set(predictions)) == 1, (
        f"identical rows produced different predictions: {predictions}"
    )


def test_first_class_higher_survival(model_data, trained_model):
    _, X_test, _, _ = model_data

    passenger = X_test.head(1).copy()

    passenger["Pclass"] = 1
    first_class = trained_model.predict_proba(passenger)[0][1]

    passenger["Pclass"] = 3
    third_class = trained_model.predict_proba(passenger)[0][1]

    assert first_class > third_class, (
        f"first class {first_class:.2%} did not beat third class {third_class:.2%}"
    )


def test_golden_predictions(model_data):
    X_train, X_test, y_train, _ = model_data

    model = train_decision_tree(X_train, y_train, max_depth=GOLDEN_MAX_DEPTH)
    predictions = list(model.predict(X_test.head(len(GOLDEN_PREDICTIONS))))

    assert predictions == GOLDEN_PREDICTIONS, (
        f"golden predictions changed: {predictions} != {GOLDEN_PREDICTIONS}"
    )


def test_no_regression(model_data, trained_model):
    _, X_test, _, y_test = model_data

    _, accuracy, _ = evaluate_model(trained_model, X_test, y_test)

    baseline = json.loads(BASELINE_JSON.read_text())
    floor = baseline["accuracy"] - REGRESSION_TOLERANCE

    assert accuracy >= floor, (
        f"accuracy {accuracy:.2%} fell below baseline {baseline['accuracy']:.2%} "
        f"minus {REGRESSION_TOLERANCE:.0%} tolerance"
    )
