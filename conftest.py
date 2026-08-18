import pytest

from src.load_data import load_data


@pytest.fixture
def titanic_df():
    return load_data()
