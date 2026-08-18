from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

actual = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
predicted = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

print(f"Accuracy:  {accuracy_score(actual, predicted):.2%}")
print(f"Precision: {precision_score(actual, predicted):.2%}")
print(f"Recall:    {recall_score(actual, predicted):.2%}")
print(f"F1 Score:  {f1_score(actual, predicted):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))
