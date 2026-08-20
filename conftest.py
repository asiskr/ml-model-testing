import pytest

from src.clean_data import clean_data
from src.config import MAX_DEPTH
from src.load_data import load_data, separate_features_target
from src.train_model import split_data, train_random_forest


@pytest.fixture
def titanic_df():
    return load_data()


@pytest.fixture
def model_data(titanic_df):
    X, y = separate_features_target(titanic_df)
    X = clean_data(X)
    return split_data(X, y)


@pytest.fixture
def trained_model(model_data):
    X_train, _, y_train, _ = model_data
    return train_random_forest(X_train, y_train, max_depth=MAX_DEPTH)
