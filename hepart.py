import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('hepatitis.csv')
data = data.dropna()
data = data.select_dtypes(include=[np.number])
scaler = StandardScaler()

# приведем данные к единому масштабу
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns = data.columns)

# вновь добавим целевую переменную



datagr = scaled_df.groupby('age').mean().T

# выведем первые два значения, чтобы убедиться в верности результата
datagr['diff'] = abs(datagr.iloc[:, 0] - datagr.iloc[:, 1])

# остается отсортировать наш датафрейм по столбцу разницы средних в нисходящем порядке
datagr = datagr.sort_values(by=['diff'], ascending=False)

# и вывести те значения (пусть их будет 10), где разница наиболее существенная
print(datagr.head(10))

