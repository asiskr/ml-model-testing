import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import RANDOM_STATE
from src.load_data import load_data, separate_features_target
from src.train_model import split_data

df = load_data()
X, y = separate_features_target(df)

# WRONG: poore data ka mean use kiya — test data ki info leak ho gayi
X_wrong = X.copy()
X_wrong["Age"] = X_wrong["Age"].fillna(X_wrong["Age"].mean())
X_wrong["Embarked"] = X_wrong["Embarked"].fillna(X_wrong["Embarked"].mode()[0])
X_wrong = pd.get_dummies(X_wrong, columns=["Sex", "Embarked"])

X_train_w, X_test_w, y_train_w, y_test_w = split_data(X_wrong, y)


# RIGHT: pehle split, phir sirf training data se clean
X_right = pd.get_dummies(X.copy(), columns=["Sex", "Embarked"])

X_train_r, X_test_r, y_train_r, y_test_r = split_data(X_right, y)

train_mean = X_train_r["Age"].mean()
X_train_r["Age"] = X_train_r["Age"].fillna(train_mean)
X_test_r["Age"] = X_test_r["Age"].fillna(train_mean)


model_w = RandomForestClassifier(random_state=RANDOM_STATE)
model_w.fit(X_train_w, y_train_w)
acc_w = accuracy_score(y_test_w, model_w.predict(X_test_w))

model_r = RandomForestClassifier(random_state=RANDOM_STATE)
model_r.fit(X_train_r, y_train_r)
acc_r = accuracy_score(y_test_r, model_r.predict(X_test_r))

print(f"WRONG way (leakage):  {acc_w:.2%}")
print(f"RIGHT way (no leak):  {acc_r:.2%}")
print(f"Difference:           {acc_w - acc_r:.2%}")
