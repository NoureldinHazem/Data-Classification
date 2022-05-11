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

    # TODO: find a better way to choose the classifier
    # TODO: Neural network
    # Training and testing classifiers
    # decisionTree(train_data, train_labels, test_data, test_labels)
    # naiveBayes(train_data, train_labels, test_data, test_labels)
    # kNearestNeighbors(train_data, train_labels, test_data, test_labels)
    # randomForest(train_data, train_labels, test_data, test_labels)
    adaBoost(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()
