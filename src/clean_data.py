import pandas as pd


def fill_missing_values(X):
    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
    return X


def encode_text_columns(X):
    X = pd.get_dummies(X, columns=["Sex", "Embarked"])
    return X


def clean_data(X):
    X = fill_missing_values(X)
    X = encode_text_columns(X)
    return X