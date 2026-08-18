from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.clean_data import clean_data
from src.load_data import explore_data, load_data, separate_features_target
from src.train_model import (
    evaluate_model,
    split_data,
    train_decision_tree,
    train_random_forest,
)


def main():
    df = load_data()
    explore_data(df)

    X, y = separate_features_target(df)
    X = clean_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n===== Decision Tree =====")
    dt_model = train_decision_tree(X_train, y_train)
    dt_predictions, dt_accuracy, dt_mae = evaluate_model(dt_model, X_test, y_test)
    print(f"Accuracy: {dt_accuracy:.2%}")
    print(f"MAE:      {dt_mae:.4f}")

    print("\n===== Random Forest =====")
    rf_model = train_random_forest(X_train, y_train)
    rf_predictions, rf_accuracy, rf_mae = evaluate_model(rf_model, X_test, y_test)
    print(f"Accuracy: {rf_accuracy:.2%}")
    print(f"MAE:      {rf_mae:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))

    print(f"\nPrecision: {precision_score(y_test, rf_predictions):.2%}")
    print(f"Recall:    {recall_score(y_test, rf_predictions):.2%}")
    print(f"F1 Score:  {f1_score(y_test, rf_predictions):.2%}")

    print(f"\nFirst 10 predictions: {rf_predictions[:10]}")
    print(f"First 10 actual:      {y_test.values[:10]}")


if __name__ == "__main__":
    main()
