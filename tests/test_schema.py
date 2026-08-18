import pandera.pandas as pa
import pytest

from src.schema import titanic_schema

pytestmark = pytest.mark.schema


def test_schema_passes_on_raw_data(titanic_df):
    titanic_schema.validate(titanic_df)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("Survived", 5),
        ("Pclass", 4),
        ("Sex", "unknown"),
        ("Age", 150.0),
        ("Fare", -1.0),
    ],
)
def test_schema_rejects_bad_value(titanic_df, column, bad_value):
    titanic_df.loc[0, column] = bad_value
    with pytest.raises(pa.errors.SchemaError):
        titanic_schema.validate(titanic_df)


def test_schema_rejects_missing_column(titanic_df):
    with pytest.raises(pa.errors.SchemaError):
        titanic_schema.validate(titanic_df.drop(columns=["Fare"]))
