from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


actual =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
predicted = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(actual, predicted)

accuracy  = accuracy_score(actual, predicted)

precision  = precision_score(actual, predicted)

recall  = recall_score(actual, predicted)

f1 = f1_score(actual, predicted)

print("Recall: ", recall , "Precsion: ", precision, "Accuracy: ", accuracy, "F1 score: ", f1, "Confusion Matrix: ", cm)