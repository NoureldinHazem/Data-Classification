from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def balanceData(dataframe):
    data, labels = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    rus = RandomUnderSampler()
    dataframe, labels_resampled = rus.fit_resample(data, labels)
    dataframe['10'] = labels_resampled
    return dataframe


def splitData(dataframe):
    data, labels = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    return X_train, X_test, y_train, y_test
