#====================================================================
# Predicción de la tasa de natalidad con redes neuronales
#====================================================================

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, regularizers

# Carga y exploración de los datos
df = pd.read_csv('dataset_natalidad.csv')

# Inspección inicial del dataset
print(f'\nPrimeras filas:\n{df.head()}\n')
print('\nTipos de datos:')
df.info()
print(f'\nEstadísticas descriptivas:\n{df.describe()}\n')
print(f'\nValores nulos:\n{df.isnull().sum()}\n')
print(f'\nCantidad de duplicados: {df.duplicated().sum()}\n')

# Selección de variables numéricas para análisis exploratorio
df_numerico = df.select_dtypes(include='number')

sns.set_palette('colorblind')

# Boxplots de variables numéricas
for col in df_numerico:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot de {col}', y=1.02)
    plt.xlabel('')
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Pairplot para relaciones entre variables numéricas
sns.pairplot(df_numerico, diag_kind='kde', height=3)
plt.suptitle('Relaciones entre variables numéricas', y=1.02)
plt.tight_layout()
plt.show()

# Matriz de correlación
correlacion = df_numerico.corr().round(2)
plt.figure(figsize=(16,10))
sns.heatmap(correlacion, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Matriz de correlación', y=1.02)
plt.tight_layout()
plt.show()

# Comentario interpretativo de correlaciones
# -> PIB per cápita (-0.87): mayor ingreso se asocia a menor tasa de natalidad.
# -> Edad de maternidad (-0.30): retraso en la maternidad se vincula a menos nacimientos.
# -> Empleo femenino y urbanización: correlaciones positivas leves.
# -> Nivel educativo y acceso a salud: baja influencia en la predicción.

# Definición de variables predictoras y objetivo
objetivo = 'Tasa_Natalidad'
X = df.drop(columns=[objetivo])
y = df[objetivo]
X = X.select_dtypes(include='number')

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de variables predictoras
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Dimensión de entrada
input_dim = X_train.shape[1]
print(f'\nNúmero de características de entrada: {input_dim}\n')

# Lista para almacenar métricas
metricas = []

# Función de cálculo de métricas de evaluación
def calculo_metricas(y_true, y_pred, act, lr):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    metricas.append({
        'Activacion': act,
        'LearningRate': lr,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
    })
    resultados = pd.DataFrame(metricas).round(3)
    print(f'\n{resultados.tail(1)}\n')
    return resultados

# Entrenamiento de redes neuronales con distintas configuraciones
for act in ['relu', 'tanh']:
    for lr in [0.01, 0.001]:
        modelo = Sequential()
        modelo.add(Dense(16, input_dim=input_dim, activation=act,
                         kernel_regularizer=regularizers.l2(1e-4)))
        modelo.add(Dense(8, activation=act,
                         kernel_regularizer=regularizers.l2(1e-4)))
        modelo.add(Dense(1, activation='linear'))

        opt = optimizers.Adam(learning_rate=lr)
        modelo.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                             validation_data=(X_test, y_test),
                             callbacks=[early_stopping], verbose=0)

        y_pred = modelo.predict(X_test).flatten()

        # Curvas de entrenamiento y validación
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title(f'Pérdida de entrenamiento vs validación ({act}, lr={lr})')
        plt.tight_layout()
        plt.legend()

        # Comparación entre valores reales y predichos
        plt.subplot(1,2,2)
        plt.scatter(y_test, y_pred, alpha=0.7)
        min_val, max_val = y_test.min(), y_test.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Tasa de natalidad real')
        plt.ylabel('Tasa de natalidad predicha')
        plt.title('Comparación entre valores reales y predichos')
        plt.legend()
        plt.tight_layout()
        plt.show()

        calculo_metricas(y_test, y_pred, act, lr)

# Resultados comparativos de todos los modelos
df_metricas = pd.DataFrame(metricas).round(3)
print(f'\nResumen comparativo de todos los modelos:\n {df_metricas}\n')

# Gráfico comparativo de R2
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.barplot(data=df_metricas, x='Activacion', y='R2', hue='LearningRate')
plt.title('Comparación de R2 según activación y learning rate')
plt.ylim(0,1)
plt.tight_layout()

# Gráfico comparativo de RMSE
plt.subplot(1,2,2)
sns.barplot(data=df_metricas, x='Activacion', y='RMSE', hue='LearningRate')
plt.title('Comparación de RMSE según activación y learning rate')
plt.tight_layout()
plt.show()

# Conclusiones
best_r2_row = df_metricas.loc[df_metricas['R2'].idxmax()]
best_rmse_row = df_metricas.loc[df_metricas['RMSE'].idxmin()]

print('\n--- CONCLUSIONES ---')
print(f'-> El mejor modelo según R2 fue con activación "{best_r2_row["Activacion"]}" y learning rate {best_r2_row["LearningRate"]}, con R2 = {best_r2_row["R2"]:.3f}.')
print(f'-> El mejor modelo según RMSE fue con activación "{best_rmse_row["Activacion"]}" y learning rate {best_rmse_row["LearningRate"]}, con RMSE = {best_rmse_row["RMSE"]:.3f}.')
print('-> Los resultados confirman la relación negativa entre PIB per cápita y natalidad y la influencia moderada de variables como edad de maternidad y empleo femenino.')
print('-> Una mejora potencial sería ampliar el dataset.')