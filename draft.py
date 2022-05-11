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
