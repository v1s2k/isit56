import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler





hepatiti_df = pd.read_csv('boston.csv')
hepatiti_df= hepatiti_df.dropna()
hepatiti_df = hepatiti_df.select_dtypes(include=[np.number])
print(hepatiti_df.head(2))

X = hepatiti_df.iloc[:, :-1] # все столбцы, кроме последнего
y = hepatiti_df.iloc[:, -1] # последний столбец
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_report = classification_report(y_train, y_train_pred)
print("Обучающая выборка:")
print(train_report)

y_test_pred= model.predict((X_test))
test_report = classification_report(y_test,y_test_pred)
print(test_report)