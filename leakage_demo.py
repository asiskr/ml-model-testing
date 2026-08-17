from load_data import load_data, separate_features_target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = load_data("train.csv")
X, y = separate_features_target(df)

# WRONG: poore data ka mean use kiya — test data ki info leak ho gayi
X_wrong = X.copy()
X_wrong["Age"] = X_wrong["Age"].fillna(X_wrong["Age"].mean())
X_wrong["Embarked"] = X_wrong["Embarked"].fillna(X_wrong["Embarked"].mode()[0])
X_wrong = pd.get_dummies(X_wrong, columns=["Sex", "Embarked"])

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wrong, y, test_size=0.2, random_state=42)



# RIGHT: pehle split, phir sirf training data se clean
X_right = X.copy()
X_right = pd.get_dummies(X_right, columns=["Sex", "Embarked"])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_right, y, test_size=0.2, random_state=42)

train_mean = X_train_r["Age"].mean()
X_train_r["Age"] = X_train_r["Age"].fillna(train_mean)
X_test_r["Age"] = X_test_r["Age"].fillna(train_mean)


# Wrong way model
model_w = RandomForestClassifier(random_state=42)
model_w.fit(X_train_w, y_train_w)
acc_w = accuracy_score(y_test_w, model_w.predict(X_test_w))

# Right way model
model_r = RandomForestClassifier(random_state=42)
model_r.fit(X_train_r, y_train_r)
acc_r = accuracy_score(y_test_r, model_r.predict(X_test_r))

print(f"WRONG way (leakage):  {acc_w:.2%}")
print(f"RIGHT way (no leak):  {acc_r:.2%}")