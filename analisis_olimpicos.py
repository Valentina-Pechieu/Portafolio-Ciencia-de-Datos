#========================================
# Análisis de datos de atletas olímpicos
#========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score

# --- Análisis Exploratorio ---
df = pd.read_csv('olimpicos.csv')

print('\n--- DataFrame (primeras filas) ---')
print(df.head())

print('\n--- Información del DataFrame ---')
print(df.info())

print('\n--- Estadísticas descriptivas ---')
print(df.describe())

# Histograma de entrenamientos
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='Entrenamientos_Semanales', bins=5, color='lightblue', edgecolor='black')
plt.title('Frecuencia de entrenamientos por semana')
plt.xlabel('Entrenamientos Semanales')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# --- Estadística Descriptiva ---
print('\n--- Tipo de variables ---')
print('Atleta: Categórica Nominal')
print('Edad: Cuantitativa Discreta')
print('Altura_cm: Cuantitativa Continua')
print('Peso_kg: Cuantitativa Continua')
print('Deporte: Categórica Nominal')
print('Entrenamientos_Semanales: Cuantitativa Discreta')
print('Medallas_Totales: Cuantitativa Discreta')
print('Pais: Categórica Nominal')

media = df['Medallas_Totales'].mean()
mediana = df['Medallas_Totales'].median()
moda = df['Medallas_Totales'].mode()[0]

print(f'\nMedia de medallas obtenidas: {media:.2f}')
print(f'Mediana de medallas obtenidas: {mediana:.2f}')
print(f'Moda de medallas obtenidas: {moda}')

desviacion = df['Altura_cm'].std()
print(f'\nDesviación estándar de la altura de los atletas: {desviacion:.2f}')

plt.figure(figsize=(5,8))
sns.boxplot(data=df, y='Peso_kg')
plt.title('Boxplot del Peso (Kg)')
plt.ylabel('Peso (Kg)')
plt.grid(True, axis='x', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Análisis de Correlación ---
matriz_corr = df.select_dtypes(include='number').corr()
valor_corr = matriz_corr.loc['Entrenamientos_Semanales', 'Medallas_Totales']

print(f'Correlación de Pearson entre Entrenamientos Semanales y Medallas Totales: {valor_corr:.2f}\n')

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Peso_kg', y='Medallas_Totales')
plt.title('Gráfico de dispersión: Peso vs Medallas Totales')
plt.xlabel('Peso (Kg)')
plt.ylabel('Medallas Totales')
plt.show()

# --- Regresión Lineal ---
x = df['Entrenamientos_Semanales']
y = df['Medallas_Totales']

X = sm.add_constant(x)  # añadimos constante para el intercepto
modelo = sm.OLS(y, X).fit()
print('\n--- Resumen del modelo de regresión ---')
print(modelo.summary())

# Interpretación de coeficientes
intercepto = modelo.params['const']
pendiente = modelo.params['Entrenamientos_Semanales']
print(f'\nInterpretación: El intercepto es {intercepto:.2f}, lo que representa las medallas esperadas cuando los entrenamientos son 0.')
print(f'La pendiente es {pendiente:.2f}, lo que indica el cambio promedio en el número de medallas por cada entrenamiento semanal adicional.')

# R2
y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)
print(f'\nEl coeficiente de determinación R2 es: {r2:.3f}, lo que indica que el modelo explica aproximadamente el {r2*100:.1f}% de la variabilidad de las medallas.\n')
print('Por lo que no es un predictor muy fuerte')
# Gráfico de regresión
plt.figure(figsize=(8,6))
sns.regplot(data=df, x='Entrenamientos_Semanales', y='Medallas_Totales',
            color='blue', line_kws={'color':'red'})
plt.title('Regresión lineal: Entrenamientos vs Medallas Totales')
plt.xlabel('Entrenamientos Semanales')
plt.ylabel('Medallas Totales')
plt.show()

# --- Visualización de Datos ---
plt.figure(figsize=(13,7))
sns.heatmap(matriz_corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de correlación')
plt.show()

plt.figure(figsize=(13,7))
sns.boxplot(data=df, y='Medallas_Totales', hue='Deporte')
plt.title('Boxplot de Medallas Totales por disciplina deportiva')
plt.ylabel('Medallas Totales')
plt.grid(True, axis='y', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()