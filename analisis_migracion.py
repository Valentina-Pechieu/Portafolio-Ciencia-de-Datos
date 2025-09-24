#================================
# Análisis de datos de migración
#================================
# Análisis de datos de migración con NumPy y Pandas

import pandas as pd
import numpy as np

# --- Carga del dataset ---
datos = pd.read_csv('migracion.csv')
df = pd.DataFrame(datos)

print('----- Limpieza y Transformación de Datos -----')
print('\nPrimeras filas del dataset:')
print(df.head())

# Conteo de valores nulos
print('\n--- Cantidad de valores nulos ---')
print(df.isnull().sum())

# --- Detección y filtrado de outliers con NumPy (IQR) ---
df_numerico = df.select_dtypes(include='number')

# Calculamos Q1 y Q3 con NumPy
q1 = np.percentile(df_numerico, 25, axis=0)
q3 = np.percentile(df_numerico, 75, axis=0)
iqr = q3 - q1
inferior = q1 - 1.5 * iqr
superior = q3 + 1.5 * iqr

# Filtramos los outliers
columnas_numericas = df_numerico.columns
df_sin_out = df.copy()
for i, columna in enumerate(columnas_numericas):
    df_sin_out = df_sin_out[(df_sin_out[columna] >= inferior[i]) &
                            (df_sin_out[columna] <= superior[i])]

# --- Reemplazo de valores en columna Razon_Migracion ---
df_sin_out.loc[df_sin_out['Razon_Migracion'] == 'Económica', 'Razon_Migracion'] = 'Trabajo'
df_sin_out.loc[df_sin_out['Razon_Migracion'] == 'Conflicto', 'Razon_Migracion'] = 'Guerra'

# Reiniciamos índices
df_sin_out = df_sin_out.reset_index(drop=True)

# --- Análisis Exploratorio ---
print('\n----- Análisis Exploratorio -----')
print(df_sin_out.head())
print('\nInformación del dataset:')
print(df_sin_out.info())
print('\nResumen estadístico:')
print(df_sin_out.describe())

# Media y mediana con NumPy
media = np.mean(df_sin_out['Cantidad_Migrantes'])
mediana = np.median(df_sin_out['Cantidad_Migrantes'])
print(f'\nLa media de migrantes es: {media:.2f} y la mediana es: {mediana:.2f}')

# PIB promedio de origen y destino
cantidad_pib_origen = df_sin_out['PIB_Origen'].value_counts()
cantidad_pib_destino = df_sin_out['PIB_Destino'].value_counts()
promedio_origen = 0
promedio_destino = 0
for i in range(len(cantidad_pib_origen)):
    o = cantidad_pib_origen.index[i] * cantidad_pib_origen.iloc[i]
    promedio_origen += o
for i in range(len(cantidad_pib_destino)):
    d = cantidad_pib_destino.index[i] * cantidad_pib_destino.iloc[i]
    promedio_destino += d
print(f'PIB promedio de países de origen: {promedio_origen}')
print(f'PIB promedio de países de destino: {promedio_destino}')

# --- Agrupamiento y Sumarización ---
print('\n----- Agrupamiento y Sumarización de Datos -----')
print('\nSuma total de migrantes por razón de migración:')
print(df_sin_out.groupby('Razon_Migracion')['Cantidad_Migrantes'].sum())

print('\nPromedio de IDH en países de origen por razón de migración:')
print(df_sin_out.groupby('Razon_Migracion')['IDH_Origen'].mean())

# Ordenamos de mayor a menor cantidad de migrantes
df_sin_out = df_sin_out.sort_values('Cantidad_Migrantes', ascending=False)
print('\nDataset ordenado por cantidad de migrantes:')
print(df_sin_out)

#Filtros y Selección de Datos
print('\n----- Filtros y Selección de Datos -----')
print('\nMigraciones por guerra:')
print(df_sin_out[df_sin_out['Razon_Migracion'] == 'Guerra'])

print('\nMigraciones con IDH Destino > 0.90:')
print(df_sin_out[df_sin_out['IDH_Destino'] > 0.90])

# Nueva columna Diferencia_IDH
df_sin_out['Diferencia_IDH'] = round(df_sin_out['IDH_Destino'] - df_sin_out['IDH_Origen'], 2)
print('\nDataset con nueva columna Diferencia_IDH:')
print(df_sin_out)

# Exportación de datos 
df_sin_out.to_csv('migracion_limpio.csv', index=False)
print('\nArchivo "Migracion_Limpio.csv" exportado correctamente.')
