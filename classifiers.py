from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def decisionTree(train_data, train_labels, test_data, test_labels):
    clf = DecisionTreeClassifier().fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    print("Decision Tree Classifier Measures:")
    report(test_labels, predicted_labels)


def report(test_labels, predicted_labels):
    accuracy = accuracy_score(test_labels, predicted_labels)
    confusionMatrix = confusion_matrix(test_labels, predicted_labels)
    TP, FP = confusionMatrix[0]
    FN, TN = confusionMatrix[1]
    print(classification_report(test_labels, predicted_labels))
    print("Accuracy:", accuracy*100, "%")
    print("Confusion Matrix:\n")
    print(f'TP:{TP}\tFP:{FP}\nFN:{FN}\tTN:{TN}\n')
    print(confusionMatrix)



