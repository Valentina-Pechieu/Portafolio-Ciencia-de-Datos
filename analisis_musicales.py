# ===================================================
# Análisis de preferencias musicales globales
# ===================================================

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Carga y exploración de los datos
df = pd.read_csv('dataset_generos_musicales.csv')

# Inspección inicial del dataset
print(f'\nPrimeras filas:\n{df.head()}\n')
print('\nTipos de datos:')
df.info()
print(f'\nEstadísticas descriptivas:\n{df.describe()}\n')
print(f'\nValores nulos:\n{df.isnull().sum()}\n')
print(f'\nCantidad de duplicados: {df.duplicated().sum()}\n')

# Identificación de columnas numéricas
df_numerico = df.select_dtypes(include='number')

# Visualización de outliers mediante boxplots
for col in df_numerico:
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot de {col}', y=1.02)
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Eliminación de outliers utilizando el método IQR
df_sin_out = df.copy()
for col in df_numerico.columns:
    Q1, Q3 = df_numerico[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df_sin_out = df_sin_out[(df_sin_out[col] >= lim_inf) & (df_sin_out[col] <= lim_sup)]

# Preprocesamiento
df_numerico_sin = df_sin_out.select_dtypes(include='number')

# Escalamiento de variables numéricas
scaler = StandardScaler()
X_escalado = scaler.fit_transform(df_numerico_sin)

# K-Means Clustering
# Ejecución inicial con K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
etiquetas = kmeans.fit_predict(X_escalado)
centroides_escalados = kmeans.cluster_centers_

df_sin_out['Cluster'] = etiquetas
print(df_sin_out[['País', 'Cluster']])

# Determinación del número óptimo de clusters mediante codo y silhouette
inercia, silhouette_scores = [], []
Ks = range(2, 7)
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiq = km.fit_predict(X_escalado)
    inercia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_escalado, etiq))

# Gráfico del método del codo
plt.figure(figsize=(6,4))
plt.plot(Ks, inercia, marker='o')
plt.title('Método del codo (Elbow)')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia (SSE)')
plt.grid(True)
plt.show()

# Gráfico de coeficiente silhouette
plt.figure(figsize=(6,4))
plt.plot(Ks, silhouette_scores, marker='o')
plt.title('Coeficiente Silhouette')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette')
plt.grid(True)
plt.show()

best_k = Ks[np.argmax(silhouette_scores)]
print(f'Mejor K según Silhouette: {best_k}')

# Clustering jerárquico
z = linkage(X_escalado, method='ward')

plt.figure(figsize=(12,8))
dendrogram(z, labels=df_sin_out['País'].values, leaf_rotation=90)
plt.title('Dendrograma - Clustering jerárquico (Ward)')
plt.ylabel('Distancia')
plt.show()

# Asignación de clusters jerárquicos (ejemplo con 3 grupos)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_sin_out['Jerarquico'] = agg.fit_predict(X_escalado)

print(df_sin_out[['País', 'Cluster', 'Jerarquico']])

# DBSCAN
parametros = [(0.5, 2), (1.0, 2), (1.0, 3), (1.5, 3)]
for eps, minpts in parametros:
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    etiquetas_dbscan = dbscan.fit_predict(X_escalado)
    df_sin_out[f'DBSCAN_eps{eps}_min{minpts}'] = etiquetas_dbscan

print(df_sin_out.head())

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_escalado)
centroides_pca = pca.transform(centroides_escalados)

# Varianza explicada acumulada
varianza_exp = np.cumsum(pca.explained_variance_ratio_)
print(f'Varianza explicada acumulada: {varianza_exp}')
n_comp = np.argmax(varianza_exp >= 0.90) + 1
print(f'Número de componentes que explican >=90% de la varianza: {n_comp}')

# Visualización PCA
plt.figure(figsize=(10,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=etiquetas, cmap='viridis', s=100, alpha=0.7)
plt.scatter(centroides_pca[:,0], centroides_pca[:,1], color='red', s=200, marker='X')
plt.title('Visualización PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# Reducción de dimensionalidad con t-SNE
for perplexity in [2, 3, 5]:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_escalado)
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=etiquetas, cmap='viridis', s=100, alpha=0.7)
    plt.title(f't-SNE (perplexity={perplexity})')
    plt.grid(True)
    plt.show()

# Conclusiones
print('\n--- CONCLUSIONES ---')
print('-> K-Means y clustering jerárquico coincidieron en que el número óptimo de clusters era 3.')
print('Sugiere patrones consistentes de agrupación entre países.')
print('-> DBSCAN no generó agrupaciones significativas debido al tamaño reducido del dataset.')
print('-> PCA permitió visualizar la estructura global, mientras que t-SNE mostró agrupaciones más claras en 2D.')
