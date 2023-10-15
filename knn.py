import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

param_grid= {
    'n_neighbors': [1, 3, 5, 7],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']} #вес соседа относительно расстояния
# Подбор параметров с помощью сетки
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Получение лучшей модели
best_model = grid_search.best_estimator_
print(best_model)

print('Train Accuracy:', accuracy_train)
print('Train Precision:', precision_train)
print('Train Recall:', recall_train)
print('Train F1-score:', f1_train)

print('Test Accuracy:', accuracy_test)
print('Test Precision:', precision_test)
print('Test Recall:', recall_test)
print('Test F1-score:', f1_test)


tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_test)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='rainbow')
plt.title('t-SNE visualization')
plt.show()

umap1 = umap.UMAP(random_state=42)
X_umap = umap1.fit_transform(X_test)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_test, cmap='rainbow')
plt.title('UMAP visualization')
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow')
plt.title('классы на основе выборок с известными метками классов')
plt.show()

y_pred = model.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title('классы с метками выставленными классификатором.')
plt.show()

print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))

