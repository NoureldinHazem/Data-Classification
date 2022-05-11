from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# TODO: check the ranges of the parameters (k, n, n, folds) in Tuning
# TODO: check if plots are required
def getParameter(train_data, train_labels, maxAcc, classifier):
    acc = cross_val_score(classifier, train_data, train_labels, cv=KFold(n_splits=10)).mean()
    if acc > maxAcc:
        return acc, True, acc
    else:
        return maxAcc, False, acc


def adaBoost(train_data, train_labels, test_data, test_labels):
    maxAcc = 0
    acclist = []

    for n in range(1, 200, 20):
        ab = AdaBoostClassifier(n_estimators=n)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, ab)
        acclist.append(acc)
        best_n = n if flag else best_n
        # print("Adaboost with best n =", n, "with accuracy =", acc)
    # pyplot.plot(range(1, 200, 20), acclist)
    # pyplot.show()

    acclist = []
    start = best_n - 10 if best_n - 10 > 0 else 1

    for n in range(start, best_n + 10):
        ab = AdaBoostClassifier(n_estimators=n)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, ab)
        best_n = n if flag else best_n
        acclist.append(acc)
        # print("Adaboost with 2 best n =", n, "with accuracy =", acc)
    # pyplot.plot((start, best_n + 10), acclist)
    # pyplot.show()

    print("Adaboost with best k =", best_n)
    ab = AdaBoostClassifier(n_estimators=best_n).fit(train_data, train_labels)
    predicted_labels = ab.predict(test_data)
    print("Adaboost Classifier Measures:")
    report(test_labels, predicted_labels)


def randomForest(train_data, train_labels, test_data, test_labels):
    maxAcc = 0
    acclist = []

    for n in range(1, 200, 20):
        rf = RandomForestClassifier(n_estimators=n)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, rf)
        acclist.append(acc)
        best_n = n if flag else best_n
        # print("Random Forest with best n =", n, "with accuracy =", acc)
    # pyplot.plot(range(1, 200, 20), acclist)
    # pyplot.show()

    acclist = []
    start = best_n - 10 if best_n - 10 > 0 else 1

    for n in range(start, best_n + 10):
        rf = RandomForestClassifier(n_estimators=n)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, rf)
        best_n = n if flag else best_n
        acclist.append(acc)
        # print("Random Forest with 2 best n =", n, "with accuracy =", acc)
    # pyplot.plot((start, best_n + 10), acclist)
    # pyplot.show()

    print("Random Forest with best k =", best_n)
    rf = RandomForestClassifier(n_estimators=best_n).fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)
    print("Random Forest Classifier Measures:")
    report(test_labels, predicted_labels)


def kNearestNeighbors(train_data, train_labels, test_data, test_labels):
    maxAcc = 0
    acclist = []

    for k in range(1, 200, 5):
        knn = KNeighborsClassifier(n_neighbors=k)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, knn)
        acclist.append(acc)
        best_k = k if flag else best_k
        print("KNN with best k =", k, "with accuracy =", acc)
    # pyplot.plot(range(1, 200, 5), acclist)
    # pyplot.show()

    acclist = []
    start = best_k - 10 if best_k - 10 > 0 else 1

    for k in range(start, best_k + 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, knn)
        best_k = k if flag else best_k
        acclist.append(acc)
        # print("KNN with best 2 k =", best_k, "with accuracy =", acc)
    # pyplot.plot((start, best_k + 10), acclist)
    # pyplot.show()

    print("KNN with best k =", best_k)
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    print("K Nearest Neighbors Classifier Measures:")
    report(test_labels, predicted_labels)


def naiveBayes(train_data, train_labels, test_data, test_labels):
    nb = GaussianNB().fit(train_data, train_labels)
    predicted_labels = nb.predict(test_data)
    print("Naive Bayes Classifier Measures:")
    report(test_labels, predicted_labels)


def decisionTree(train_data, train_labels, test_data, test_labels):
    dt = DecisionTreeClassifier().fit(train_data, train_labels)
    predicted_labels = dt.predict(test_data)
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
