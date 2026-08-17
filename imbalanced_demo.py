from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

actual = [0]*90 + [1]*10
predicted = [0]*100

print(f"Accuracy:  {accuracy_score(actual, predicted):.2%}")
print(f"Precision: {precision_score(actual, predicted, zero_division=0):.2%}")
print(f"Recall:    {recall_score(actual, predicted):.2%}")
print(f"F1 Score:  {f1_score(actual, predicted):.2%}")