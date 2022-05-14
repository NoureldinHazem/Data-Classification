from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


# tuning the functions using GridSearchCV
def tuning(train_data, train_labels, classifier, parameters, hyperparameter):
    clf = GridSearchCV(classifier, parameters, cv=StratifiedKFold(n_splits=10))
    clf.fit(train_data, train_labels)
    bestValue = clf.best_params_[hyperparameter]
    return bestValue


# function to get the highest accuracy using cross validation
def getParameter(train_data, train_labels, maxAcc, classifier):
    acc = cross_val_score(classifier, train_data, train_labels, cv=StratifiedKFold(n_splits=10)).mean()
    if acc > maxAcc:
        return acc, True, acc
    else:
        return maxAcc, False, acc


# initialize the adaboost classifier
def adaBoost(train_data, train_labels, test_data, test_labels):
    # hyperparameter tuning (n_estimators)

    # maxAcc = 0
    # acclist = []
    # best_n_range = 1
    #
    # # tuning in the range of 1 to 401 with step size of 50
    # for n in range(1, 401, 50):
    #     ab = AdaBoostClassifier(n_estimators=n)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, ab)
    #     acclist.append(acc)
    #     best_n_range = n if flag else best_n_range
    #     # print("Adaboost with best n =", n, "with accuracy =", acc)
    # # Plot the accuracy vs n_estimators
    # pyplot.plot(range(1, 401, 50), acclist)
    # pyplot.title('adaboost with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    #
    # acclist = []
    # start = best_n_range - 10 if best_n_range - 10 > 0 else 1
    # best_n = best_n_range
    #
    # # tuning in the range of best_k_range to best_k_range + 10 with step size of 2
    # for n in range(start, best_n_range + 10, 2):
    #     ab = AdaBoostClassifier(n_estimators=n)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, ab)
    #     best_n = n if flag else best_n
    #     acclist.append(acc)
    #     # print("Adaboost with 2 best n =", n, "with accuracy =", acc)
    # # Plot the accuracy vs n_estimators
    # pyplot.plot(range(start, best_n_range + 10, 2), acclist)
    # pyplot.title('adaboost with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    #
    # print("Adaboost with best k =", best_n)

    # Testing on the test data with the tuned hyper parameter (n_estimator = 295)
    ab = AdaBoostClassifier(n_estimators=295).fit(train_data, train_labels)
    predicted_labels = ab.predict(test_data)
    print("Adaboost Classifier Measures:")
    report(test_labels, predicted_labels)


# initialize the random forest classifier
def randomForest(train_data, train_labels, test_data, test_labels):
    # hyperparameter tuning (n_estimators)

    # maxAcc = 0
    # acclist = []
    # best_n_range = 1
    #
    # # tuning in the range of 3 to 403 with step size of 50
    # for n in range(3, 403, 50):
    #     rf = RandomForestClassifier(n_estimators=n)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, rf)
    #     acclist.append(acc)
    #     best_n_range = n if flag else best_n_range
    #
    # # Plot the accuracy vs n_estimators
    # pyplot.plot(range(3, 403, 50), acclist)
    # pyplot.title('Random Forest with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    #
    # acclist = []
    # start = best_n_range - 10 if best_n_range - 10 > 0 else 1
    # best_n = best_n_range
    #
    # # tuning in the range of best_k_range to best_k_range + 10 with step size of 2
    # for n in range(start, best_n_range + 10, 2):
    #     rf = RandomForestClassifier(n_estimators=n)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, rf)
    #     best_n = n if flag else best_n
    #     acclist.append(acc)
    #
    # # Plot the accuracy vs n_estimators
    # pyplot.plot(range(start, best_n_range + 10, 2), acclist)
    # pyplot.title('Random Forest with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    #
    # print("Random Forest with best k =", best_n)

    # Testing on the test data with the tuned hyper parameter (n_estimator = 195)
    rf = RandomForestClassifier(n_estimators=195).fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)
    print("Random Forest Classifier Measures:")
    report(test_labels, predicted_labels)


# initialize the KNN classifier
def kNearestNeighbors(train_data, train_labels, test_data, test_labels):
    # hyperparameter tuning (K)

    # maxAcc = 0
    # acclist = []
    # best_k_range = 1
    # for k in range(1, 200, 10):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, knn)
    #     acclist.append(acc)
    #     best_k_range = k if flag else best_k_range
    #
    # # Plot the accuracy vs K
    # pyplot.plot(range(1, 200, 10), acclist)
    # pyplot.title('KNN')
    # pyplot.xlabel("K")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    #
    # acclist.clear()
    # start = best_k_range - 10 if best_k_range - 10 > 0 else 1
    # best_k = best_k_range
    #
    # tuning in the range of best_k_range to best_k_range + 10
    # for k in range(start, best_k_range + 10):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     maxAcc, flag, acc = getParameter(train_data, train_labels, maxAcc, knn)
    #     best_k = k if flag else best_k
    #     acclist.append(acc)
    #
    # # Plot the accuracy vs K
    # pyplot.plot(range(start, best_k_range + 10), acclist)
    # pyplot.title('KNN')
    # pyplot.xlabel("K")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    # print("KNN with best k =", best_k)

    # Testing on the test data with the tuned hyper parameter (K = 9)
    knn = KNeighborsClassifier(n_neighbors=9).fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    print("K Nearest Neighbors Classifier Measures:")
    report(test_labels, predicted_labels)


# initialize the Naive Bayes classifier
def naiveBayes(train_data, train_labels, test_data, test_labels):
    nb = GaussianNB().fit(train_data, train_labels)
    predicted_labels = nb.predict(test_data)
    print("Naive Bayes Classifier Measures:")
    report(test_labels, predicted_labels)


# initialize the Decision tree classifier
def decisionTree(train_data, train_labels, test_data, test_labels):
    dt = DecisionTreeClassifier().fit(train_data, train_labels)
    predicted_labels = dt.predict(test_data)
    print("Decision Tree Classifier Measures:")
    report(test_labels, predicted_labels)


# print measures for each classifier
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
