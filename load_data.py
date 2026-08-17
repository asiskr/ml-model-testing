import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    print("--- First 5 Rows ---")
    print(df.head())
    print("\n--- Data Info ---")
    print(df.info())
    print("\n--- Statistics ---")
    print(df.describe())


def separate_features_target(df):
    y = df["Survived"]
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    return X, y