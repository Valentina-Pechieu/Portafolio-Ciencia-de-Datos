#====================================================================
# Análisis del impacto del cambio climático en la agricultura
#====================================================================
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Carga y exploración de datos
df = pd.read_csv('cambio_climatico_agricultura.csv')

# Inspección inicial del dataset
print(f'\nPrimeras filas:\n{df.head()}\n')
print('\nTipos de datos:')
df.info()
print(f'\nEstadísticas descriptivas:\n{df.describe()}\n')
print(f'\nValores nulos:\n{df.isnull().sum()}\n')
print(f'\nCantidad de duplicados: {df.duplicated().sum()}\n')

# Visualización exploratoria
df_numerico = df.select_dtypes(include='number')
sns.set_palette('colorblind')

# Pairplot de relaciones entre variables numéricas
sns.pairplot(df_numerico, diag_kind='kde', height=3)
plt.suptitle('Relaciones entre variables numéricas', y=1.02)
plt.show()

# Boxplots de cada variable numérica para detectar outliers
for col in df_numerico:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot de {col}', y=1.02)
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Matriz de correlación
correlacion = df_numerico.corr().round(2)
plt.figure(figsize=(10,5))
sns.heatmap(correlacion, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Matriz de correlación', y=1.02)
plt.tight_layout()
plt.show()

# Preprocesamiento y escalamiento
target = 'Producción_alimentos'
X = df.drop(columns=[target])  # Variables predictoras
y = df[target]                 # Variable objetivo

# Identificación de variables numéricas y categóricas
columna_numerica_reg = X.select_dtypes(include='number').columns.tolist()
columna_categorica_reg = X.select_dtypes(exclude='number').columns.tolist()

# Definición de pipelines de transformación
numerica_pipe = Pipeline([('sc', StandardScaler())])  
categorica_pipe = Pipeline([('oh', OneHotEncoder(handle_unknown='ignore'))])  

# ColumnTransformer para aplicar transformaciones
pre_reg = ColumnTransformer(
    transformers=[
        ('num', numerica_pipe, columna_numerica_reg),
        ('cat', categorica_pipe, columna_categorica_reg),
    ],
    remainder='drop'
)

# División de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de regresión
metricas_reg = []
def calculo_metricas_reg(y_test, y_pred, nombre_modelo):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metricas_reg.append({
        'Modelo_Regresion': nombre_modelo,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

# Definición de modelos de regresión
pipe_reg = Pipeline([('pre', pre_reg), ('reg', LinearRegression())])
pipe_tree_reg = Pipeline([('pre', pre_reg), ('reg', DecisionTreeRegressor(max_depth=3, random_state=42))])
pipe_rf_reg = Pipeline([('pre', pre_reg), ('reg', RandomForestRegressor(n_estimators=300, random_state=42))])

# Entrenamiento de modelos básicos
for nombre, modelo in [('Regresion_Lineal', pipe_reg),
                       ('Arbol_Reg', pipe_tree_reg),
                       ('RandomForest_Reg', pipe_rf_reg)]:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    calculo_metricas_reg(y_test, y_pred, nombre)

# Regularización con Ridge y Lasso mediante GridSearchCV
ridge_param_grid = {'reg__alpha': [0.01, 0.1, 1, 10, 100]}
lasso_param_grid = {'reg__alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
cv = KFold(n_splits=3, shuffle=True, random_state=42)

ridge_pipe = Pipeline([('pre', pre_reg), ('reg', Ridge(random_state=42))])
lasso_pipe = Pipeline([('pre', pre_reg), ('reg', Lasso())])

for nombre, modelo, param_grid in [('Ridge', ridge_pipe, ridge_param_grid),
                                   ('Lasso', lasso_pipe, lasso_param_grid)]:
    grid = GridSearchCV(modelo, param_grid, cv=cv,
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred_best = best.predict(X_test)
    calculo_metricas_reg(y_test, y_pred_best, nombre)

# Resultados de regresión
df_metricas_reg = pd.DataFrame(metricas_reg).set_index('Modelo_Regresion')
print('\n--- Resultados de Regresión ---')
print(df_metricas_reg)

# Modelos de clasificación
# Creación de variable categórica de impacto climático (bajo, medio, alto)
climatico = pd.qcut(df['Producción_alimentos'], q=3, labels=['Bajo','Medio','Alto'])
df['Impacto_Climático'] = climatico

metricas_clf = []
def calculo_metricas_clf(y_test, y_pred, y_pred_proba, nombre_modelo):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    metricas_clf.append({
        'Modelo_Clasificacion': nombre_modelo,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc
    })

# Variables predictoras y target para clasificación
X_clf = df[['Temperatura_promedio','Cambio_lluvias','Frecuencia_sequías','País']]
y_clf = df['Impacto_Climático']

# Identificación de variables categóricas y numéricas
columna_categorica_clf = X_clf.select_dtypes(exclude='number').columns.tolist()
columna_numerica_clf = X_clf.select_dtypes(include='number').columns.tolist()

print(f'\nConteo de clases:\n{y_clf.value_counts()}')
print(f'\nProporciones (%):\n{(y_clf.value_counts(normalize=True)*100).round(0)}\n')

# Preprocesamiento para clasificación
pre_clf = ColumnTransformer(
    transformers=[
        ('num', numerica_pipe, columna_numerica_clf),
        ('cat', categorica_pipe, columna_categorica_clf),
    ],
    remainder='drop'
)

# Definición de modelos de clasificación
pipe_knn = Pipeline([('pre', pre_clf), ('clf', KNeighborsClassifier(n_neighbors=5))])
pipe_tree_clf = Pipeline([('pre', pre_clf), ('clf', DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced'))])
pipe_svm_clf = Pipeline([('pre', pre_clf), ('clf', SVC(kernel='rbf', gamma='scale', probability=True, random_state=42, class_weight='balanced'))])

# División train/test con estratificación
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf)

# Entrenamiento de modelos de clasificación
for nombre, modelo in [('KNN', pipe_knn),
                       ('Arbol_clf', pipe_tree_clf),
                       ('SVM', pipe_svm_clf)]:
    modelo.fit(X_train_clf, y_train_clf)
    y_pred_clf = modelo.predict(X_test_clf)
    y_pred_proba_clf = modelo.predict_proba(X_test_clf)

    # Matriz de confusión
    mc = confusion_matrix(y_test_clf, y_pred_clf)
    plt.figure(figsize=(8,5))
    sns.heatmap(mc, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Clase real')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión {nombre}')
    plt.show()

    calculo_metricas_clf(y_test_clf, y_pred_clf, y_pred_proba_clf, nombre)

# Optimización de modelos mediante GridSearchCV
param_grid_knn = {'clf__n_neighbors': [3, 5, 7], 'clf__weights': ['uniform', 'distance']}
param_grid_tree = {'clf__max_depth': [2, 3, 4, 5], 'clf__min_samples_split': [2, 4, 6]}
param_grid_svm = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf'], 'clf__gamma': ['scale', 'auto']}

cv1 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

for nombre, modelo, param_grid in [('KNN(Grid)', pipe_knn, param_grid_knn),
                                   ('Arbol_clf(Grid)', pipe_tree_clf, param_grid_tree),
                                   ('SVM(Grid)', pipe_svm_clf, param_grid_svm)]:
    grid_clf = GridSearchCV(modelo, param_grid, cv=cv1,
                            scoring='f1_weighted', n_jobs=-1)
    grid_clf.fit(X_train_clf, y_train_clf)
    best_clf = grid_clf.best_estimator_

    # Predicciones del mejor modelo
    y_pred_best_clf = best_clf.predict(X_test_clf)
    y_pred_proba_best_clf = best_clf.predict_proba(X_test_clf)

    print(f'Mejores parámetros {nombre}: {grid_clf.best_params_}')

    # Matriz de confusión del mejor modelo
    mc = confusion_matrix(y_test_clf, y_pred_best_clf)
    plt.figure(figsize=(8,5))
    sns.heatmap(mc, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Clase real')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión {nombre}')
    plt.show()

    calculo_metricas_clf(y_test_clf, y_pred_best_clf, y_pred_proba_best_clf, nombre)

# Resultados de clasificación
df_metricas_clf = pd.DataFrame(metricas_clf).set_index('Modelo_Clasificacion')
print('\n--- Resultados de Clasificación ---')
print(df_metricas_clf)

# Conclusiones 
best_reg = df_metricas_reg['R2'].idxmax()
best_clf = df_metricas_clf['F1'].idxmax()

print('\n--- CONCLUSION ---')
print(f'El mejor modelo de Regresión fue "{best_reg}" con R2 = {df_metricas_reg.loc[best_reg,"R2"]:.3f}.')
print(f'El mejor modelo de Clasificación fue "{best_clf}" con F1 = {df_metricas_clf.loc[best_clf,"F1"]:.3f}.')
print('Estos resultados muestran un equilibrio adecuado entre precisión y recall para predecir impacto climático.')
