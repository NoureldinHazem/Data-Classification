import csv
import numpy as np
import pandas as pd

with open('magic04.data') as file:
    df = pd.DataFrame()
    fileReader = csv.reader(file)
    for line in fileReader:
        df = pd.concat([df, pd.DataFrame(np.array(line).reshape(-1, len(line)))], ignore_index=True)
    print(df)
