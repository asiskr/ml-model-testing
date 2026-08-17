from load_data import load_data, separate_features_target
from clean_data import clean_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = load_data("train.csv")
X, y = separate_features_target(df)
X = clean_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

d_model = DecisionTreeClassifier(max_depth=3, random_state=42)
d_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)



dt_train_pred = d_model.predict(X_train)
dt_test_pred = d_model.predict(X_test)

dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

print("Decision Tree (max_depth=3):")
print(f"Training Accuracy: {dt_train_acc:.2%}")
print(f"Testing Accuracy:  {dt_test_acc:.2%}")
print(f"Gap:               {dt_train_acc - dt_test_acc:.2%}")




rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

print("\nRandom Forest:")
print(f"Training Accuracy: {rf_train_acc:.2%}")
print(f"Testing Accuracy:  {rf_test_acc:.2%}")
print(f"Gap:               {rf_train_acc - rf_test_acc:.2%}")