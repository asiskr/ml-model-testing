from sklearn.metrics import accuracy_score

from src.clean_data import clean_data
from src.load_data import load_data, separate_features_target
from src.train_model import split_data, train_decision_tree, train_random_forest

df = load_data()
X, y = separate_features_target(df)
X = clean_data(X)
X_train, X_test, y_train, y_test = split_data(X, y)

d_model = train_decision_tree(X_train, y_train, max_depth=3)
rf_model = train_random_forest(X_train, y_train)

dt_train_acc = accuracy_score(y_train, d_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, d_model.predict(X_test))

print("Decision Tree (max_depth=3):")
print(f"Training Accuracy: {dt_train_acc:.2%}")
print(f"Testing Accuracy:  {dt_test_acc:.2%}")
print(f"Gap:               {dt_train_acc - dt_test_acc:.2%}")

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))

print("\nRandom Forest:")
print(f"Training Accuracy: {rf_train_acc:.2%}")
print(f"Testing Accuracy:  {rf_test_acc:.2%}")
print(f"Gap:               {rf_train_acc - rf_test_acc:.2%}")
