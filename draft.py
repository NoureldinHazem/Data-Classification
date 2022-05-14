import csv
import numpy as np
import pandas as pd

with open('magic04.data') as file:
    df = pd.DataFrame()
    fileReader = csv.reader(file)
    for line in fileReader:
        df = pd.concat([df, pd.DataFrame(np.array(line).reshape(-1, len(line)))], ignore_index=True)
    print(df)

# def tuneKNN(train_data, train_labels):
#     maxAcc = 0
#     for k in range(1, 200, 5):
#         # print("Tuning KNN with first k =", k)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         acc = cross_val_score(knn, train_data, train_labels, cv=KFold(n_splits=10)).mean()
#         if acc > maxAcc:
#             maxAcc = acc
#             best_k = k
#     start = best_k - 10 if best_k - 10 > 0 else 1
#     for k in range(start, best_k + 10):
#         # print("KNN with second k =", k)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         acc = cross_val_score(knn, train_data, train_labels, cv=KFold(n_splits=10)).mean()
#         if acc > maxAcc:
#             maxAcc = acc
#             best_k = k
#     return best_k


# Tuning KNN
parameters = {'n_neighbors': range(1, 200, 10)}
knn = KNeighborsClassifier()
best_k_range = tuning(train_data, train_labels, parameters, knn, 'n_neighbors')
start = best_k_range - 10 if best_k_range - 10 > 0 else 1
parameters = {'n_neighbors': range(start, best_k_range + 10)}
best_k = tuning(train_data, train_labels, parameters, knn, 'n_neighbors')

# Tuning Random Forest
parameters = {'n_estimators': range(1, 400, 50)}
rf = RandomForestClassifier()
best_n_range = tuning(train_data, train_labels, parameters, rf, 'n_estimators')
start = best_n_range - 10 if best_n_range - 10 > 0 else 1
parameters = {'n_estimators': range(start, best_n_range + 10)}
best_n = tuning(train_data, train_labels, parameters, rf, 'n_estimators')

# Tuning adaboost
parameters = {'n_estimators': range(1, 400, 50)}
ab = AdaBoostClassifier()
best_n_range = tuning(train_data, train_labels, parameters, ab, 'n_estimators')
print('Best R1 = ', best_n_range)
start = best_n_range - 10 if best_n_range - 10 > 0 else 1
parameters = {'n_estimators': range(start, best_n_range + 10)}
best_r = tuning(train_data, train_labels, parameters, ab, 'n_estimators')
