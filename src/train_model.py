from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, TEST_SIZE


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train, max_depth=None):

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, max_depth=None):

    model = RandomForestClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mae = mean_absolute_error(y_test, predictions)

    return predictions, accuracy, mae