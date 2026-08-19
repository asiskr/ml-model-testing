import pytest
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

from src.load_data import load_data, separate_features_target
from src.clean_data import clean_data
from src.train_model import (
    split_data,
    train_random_forest,
    train_decision_tree,
    evaluate_model
)


@pytest.fixture
def model_data():
    df = load_data("data/train.csv")

    X, y = separate_features_target(df)
    X = clean_data(X)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

@pytest.mark.model
def test_model_accuracy_above_threshold(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    predictions, accuracy, mae = evaluate_model(
        model,
        X_test,
        y_test
    )

    assert accuracy >= 0.75

@pytest.mark.model
def test_model_not_overfitting(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    _, accuracy_on_train, _ = evaluate_model(
        model,
        X_train,
        y_train
    )

    _, accuracy_on_test, _ = evaluate_model(
        model,
        X_test,
        y_test
    )

    gap = accuracy_on_train - accuracy_on_test

    assert gap < 0.15

@pytest.mark.model
def test_model_deterministic(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    prediction_1 = model.predict(X_test)
    prediction_2 = model.predict(X_test)

    assert (prediction_1 == prediction_2).all()

@pytest.mark.model
def test_single_row_prediction(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    single_row = X_test.head(1)

    predictions = model.predict(single_row)

    assert len(predictions) == 1

@pytest.mark.model
def test_prediction_shape(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})

@pytest.mark.model
def test_model_rejects_wrong_columns(model_data):
    X_train, X_test, y_train, y_test = model_data

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    bad_data = X_test.drop(columns=["Age"])

    with pytest.raises(Exception):
        model.predict(bad_data)

@pytest.mark.model
def test_random_forest_beats_decision_tree(model_data):
    X_train, X_test, y_train, y_test = model_data

    random_forest = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    decision_tree = train_decision_tree(
        X_train,
        y_train,
        max_depth=3
    )

    predictions_rf = random_forest.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predictions_rf)

    predictions_dt = decision_tree.predict(X_test)
    accuracy_dt = accuracy_score(y_test, predictions_dt)

    assert accuracy_rf >= accuracy_dt

@pytest.mark.model
def test_duplicate_row_same_prediction(model_data):
    X_train, X_test, y_train, y_test = model_data
    model = train_random_forest(X_train, y_train, max_depth=3)
    
    one_passenger = X_test.head(1)
    
    duplicated = pd.concat([one_passenger, one_passenger, one_passenger])
    
    predictions = model.predict(duplicated)
    
    assert predictions[0] == predictions[1] == predictions[2]

@pytest.mark.model
def test_first_class_higher_survival(model_data):
    X_train, X_test, y_train, y_test = model_data
    model = train_random_forest(X_train, y_train, max_depth=3)

    passengar = X_test.head(1).copy()

    passengar["Pclass"] = 1
    probability1 = model.predict_proba(passengar)[0][1]

    passengar["Pclass"] = 3
    probability3 = model.predict_proba(passengar)[0][1]

    assert probability1>probability3

@pytest.mark.model
def test_golden_predictions(model_data):
    X_train, X_test, y_train, y_test = model_data
    model = train_decision_tree(X_train, y_train, max_depth=5)
    
    first_five = X_test.head(5)
    predictions = model.predict(first_five)
    print(predictions)
    assert list(predictions) == [0, 0, 0, 1, 1]

@pytest.mark.model
def test_no_regression(model_data):

    X_train, X_test, y_train, y_test = model_data
    model = train_random_forest(X_train, y_train, max_depth=3)

    predictions_rf = model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    with open("baseline_model.pkl", "rb")as f:
        baseline = pickle.load(f)
        baseline_accuracy = baseline["accuracy"]

    assert accuracy_rf >= baseline_accuracy - 0.05