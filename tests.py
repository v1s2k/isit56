import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

cancer_df = pd.DataFrame(cancer.data,columns=cancer.feature_names)


scaler=StandardScaler()

scaled_data =scaler.fit_transform(cancer_df)

scaled_data=pd.DataFrame(scaled_data,columns=cancer.feature_names)

scaled_data['target']= cancer.target

data=scaled_data.groupby('target').mean().T

data ['diff'] = abs(data.iloc[:,0]-data.iloc[:,1])
data=data.sort_values(by=['diff'],ascending=False)

print(data.head(10))


plt.hist(scaled_data.loc[cancer.target==0,'worst concave points'],15,alpha=0.5,label='Злокачественная')
plt.hist(scaled_data.loc[cancer.target==1,'worst concave points'],15,alpha=0.5,label='Доброкачественная')

plt.xlabel('worst concave points')
plt.ylabel('Количество наблюдаемых людей')


features = list(data.index[:10])


X= scaled_data[features]
y = scaled_data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)


model = LogisticRegression()

# обучим нашу модель
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# передадим ей тестовые и прогнозные значения
# поменяем порядок так, чтобы злокачественные опухоли были положительным классом
model_matrix = confusion_matrix(y_test, y_pred, labels = [1,0])

# для удобства создадим датафрейм
model_matrix_df = pd.DataFrame(model_matrix)
model_matrix_df

# добавим подписи к столбцам и строкам через параметры columns и index
# столбец - это прогноз, строка - фактическое значение
# 0 - добр. образование, 1 - злок. образование (только в рамках матрицы ошибок!)
model_matrix_df = pd.DataFrame(model_matrix, columns = ['Прогноз добр.', 'Прогноз злок.'], index = ['Факт добр.', 'Факт злок.'])
print(model_matrix_df)

from sklearn.metrics import accuracy_score

model_accuracy = accuracy_score(y_test, y_pred)
print(round(model_accuracy, 2))