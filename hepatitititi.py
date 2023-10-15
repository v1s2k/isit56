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
print(data.head(13))

X = data[['sgot','albumin']]
y = data['age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC()
model.fit(X_train, y_train)

param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(model, param_grid, cv=2)
grid_search.fit(X_train, y_train)
best_classifier = grid_search.best_estimator_
print(best_classifier)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

recall_train = recall_score(y_train, y_train_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')

precision_train = precision_score(y_train, y_train_pred, average='weighted')
precision_test = precision_score(y_test, y_test_pred, average='weighted')

f1_train = f1_score(y_train, y_train_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')
num_support_vectors = best_classifier.support_vectors_.shape[0]
print('Train Accuracy:', accuracy_train)
print('Train Precision:', precision_train)
print('Train Recall:', recall_train)
print('Train F1-score:', f1_train)

print('Test Accuracy:', accuracy_test)
print('Test Precision:', precision_test)
print('Test Recall:', recall_test)
print('Test F1-score:', f1_test)

print(f"Число опорных векторов: {num_support_vectors}")

tsne = TSNE(n_components=2)
umap_ = umap.UMAP(random_state=42)
X_tsne = tsne.fit_transform(X_train)
X_umap = umap_.fit_transform(X_train)

plt.figure(figsize=(14, 10))
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap=plt.cm.get_cmap("Set1", 2))
plt.title("t-SNE Visualization")
plt.show()


plt.figure(figsize=(14, 10))
plt.subplot(1, 2, 2)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_train, cmap=plt.cm.get_cmap("Set1", 2))
plt.title("UMAP Visualization")
plt.show()




support_vectors = grid_search.best_estimator_.support_vectors_
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='black', marker='x')
plt.title('Support Vectors')
plt.show()


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
plt.title('классы на основе выборок с известными метками классов')
plt.show()

y_pred = model.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title('классы с метками выставленными классификатором.')
plt.show()
print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))
