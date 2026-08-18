from src.load_data import load_data


def missing_percentage(df):
    return df.isnull().mean() * 100


def duplicate_count(df):
    return df.duplicated().sum()


def value_ranges(df):
    return df.describe().loc[["min", "max"]]


def zero_fare_rows(df):
    return df[df["Fare"] == 0]


def report(df):
    print("--- Missing Values (%) ---")
    print(missing_percentage(df))
    print("\n--- Duplicate Rows ---")
    print(duplicate_count(df))
    print("\n--- Min / Max ---")
    print(value_ranges(df))
    print("\n--- Zero Fare Rows ---")
    print(zero_fare_rows(df))


if __name__ == "__main__":
    report(load_data())
