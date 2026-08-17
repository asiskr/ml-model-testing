from load_data import load_data, explore_data, separate_features_target
from clean_data import clean_data
from train_model import split_data, train_decision_tree, train_random_forest, evaluate_model


df = load_data("train.csv")
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

print(f"\nFirst 10 predictions: {rf_predictions[:10]}")
print(f"First 10 actual:      {y_test.values[:10]}")