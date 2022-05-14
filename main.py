from classifiers import *
import pandas as pd
from data import *


def main():
    # Load data
    df = pd.read_csv('magic04.data', header=None)

    # Balance data
    df = balanceData(df)

    # Split data into training and test sets
    train_data, test_data, train_labels, test_labels = splitData(df)

    # Training and testing classifiers
    print("Choose Classifier: ")
    print("1. Decision Tree")
    print("2. KNN")
    print("3. Naive Bayes")
    print("4. Random Forest")
    print("5. adaboost")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        decisionTree(train_data, test_data, train_labels, test_labels)
    elif choice == 2:
        kNearestNeighbors(train_data, train_labels, test_data, test_labels)
    elif choice == 3:
        naiveBayes(train_data, train_labels, test_data, test_labels)
    elif choice == 4:
        randomForest(train_data, train_labels, test_data, test_labels)
    elif choice == 5:
        adaBoost(train_data, train_labels, test_data, test_labels)
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
