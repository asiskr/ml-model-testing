import pandas as pd

from src.config import FEATURES, TARGET, TRAIN_CSV


def load_data(filepath=TRAIN_CSV):
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
    y = df[TARGET]
    X = df[FEATURES].copy()
    return X, y
