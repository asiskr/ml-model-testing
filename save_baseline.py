import pickle

from src.load_data import load_data, separate_features_target
from src.clean_data import clean_data
from src.train_model import (
    split_data,
    train_random_forest,
    evaluate_model
)


def baseline():

    df = load_data("data/train.csv")

    X, y = separate_features_target(df)

    X = clean_data(X)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_random_forest(
        X_train,
        y_train,
        max_depth=3
    )

    _, accuracy_on_test, _ = evaluate_model(
        model,
        X_test,
        y_test
    )

    with open("baseline_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "accuracy": accuracy_on_test
            },
            f
        )


baseline()