import pytest

from src.config import FEATURES, TARGET
from src.data_quality import duplicate_count, missing_percentage, zero_fare_rows

pytestmark = pytest.mark.quality

EXPECTED_ROWS = 891
MAX_AGE_MISSING_PCT = 25


def test_expected_columns_are_present(titanic_df):
    assert set(FEATURES + [TARGET]).issubset(titanic_df.columns)


def test_row_count_is_unchanged(titanic_df):
    assert len(titanic_df) == EXPECTED_ROWS


def test_no_duplicate_rows(titanic_df):
    assert duplicate_count(titanic_df) == 0


def test_passenger_id_is_unique(titanic_df):
    assert titanic_df["PassengerId"].is_unique


def test_target_has_no_missing_values(titanic_df):
    assert missing_percentage(titanic_df)[TARGET] == 0


def test_age_missing_stays_below_threshold(titanic_df):
    assert missing_percentage(titanic_df)["Age"] < MAX_AGE_MISSING_PCT


def test_fare_is_never_negative(titanic_df):
    assert (titanic_df["Fare"] >= 0).all()


def test_known_zero_fare_rows(titanic_df):
    assert len(zero_fare_rows(titanic_df)) == 15

def test_pclass_only_valid_values(titanic_df):

    assert titanic_df["Pclass"].isin([1,2,3]).all()
    
