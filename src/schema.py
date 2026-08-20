import pandera.pandas as pa

from src.load_data import load_data

titanic_schema = pa.DataFrameSchema(
    {
        "Survived": pa.Column(int, pa.Check.isin([0, 1])),
        "Pclass": pa.Column(int, pa.Check.isin([1, 2, 3])),
        "Sex": pa.Column(str, pa.Check.isin(["male", "female"])),
        "Age": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 120)),
        "Fare": pa.Column(float, checks=pa.Check.ge(0)),
        "Embarked": pa.Column(str, nullable=True, checks=pa.Check.isin(["S", "C", "Q"])),
    }
)


def validate(df):
    return titanic_schema.validate(df)


if __name__ == "__main__":
    validate(load_data())
    print("Schema validation passed!")
