#===============================================
# Análisis experimental del programa de tutoría
#===============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t
import math

# Diseño del experimento
grupo_a = np.array([85,90,78,88,92,80,86,89,84,87,91,82,83,85,88])  # Grupo con tutoría
grupo_b = np.array([70,72,75,78,80,68,74,76,79,77,73,71,75,78,80])  # Grupo de control

# Estadísticas descriptivas
media_a = np.mean(grupo_a)
desv_a = np.std(grupo_a, ddof=1)
media_b = np.mean(grupo_b)
desv_b = np.std(grupo_b, ddof=1)

print(f'Grupo A (Tutoría) -> Media: {media_a:.2f}. Desviación Estándar: {desv_a:.2f}')
print(f'Grupo B (Control) -> Media: {media_b:.2f}. Desviación Estándar: {desv_b:.2f}\n')

# Boxplots comparativos
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.boxplot(y=grupo_a, color='deepskyblue')
plt.title('Boxplot del Grupo A (Tutoría)')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.subplot(1,2,2)
sns.boxplot(y=grupo_b, color='violet')
plt.title('Boxplot del Grupo B (Control)')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# Histogramas comparativos
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.histplot(grupo_a, bins=5, kde=True, color='deepskyblue', edgecolor='black')
plt.title('Histograma del Grupo A (Tutoría)')
plt.xlabel('Notas de Examen')
plt.ylabel('Frecuencia')

plt.subplot(1,2,2)
sns.histplot(grupo_b, bins=5, kde=True, color='violet', edgecolor='black')
plt.title('Histograma del Grupo B (Control)')
plt.xlabel('Notas de Examen')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Prueba de hipótesis
# H0: μA = μB
# H1: μA > μB

alfa = 0.05
t_stat, p_valor_bilateral = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)

if t_stat > 0:
    p_valor_uni = p_valor_bilateral / 2
else:
    p_valor_uni = 1 - (p_valor_bilateral / 2)

print(f't-Estadístico: {t_stat:.4f}')
print(f'p-valor (unilateral): {p_valor_uni:.4f}')

if p_valor_uni < alfa:
    conclusion_test = 'Se rechaza H0. El grupo con tutoría tiene un rendimiento significativamente mayor.'
else:
    conclusion_test = 'No se rechaza H0. No hay evidencia suficiente de que la tutoría mejore el rendimiento.'

print(f'Conclusión: {conclusion_test}\n')

# Intervalo de confianza
n = 15
confianza = 0.95
s_a = np.var(grupo_a, ddof=1)
s_b = np.var(grupo_b, ddof=1)

diff_medias = media_a - media_b
s = np.sqrt(s_a/n + s_b/n)

gl = (s_a/n + s_b/n)**2 / ((s_a**2)/(n**2*(n-1)) + (s_b**2)/(n**2*(n-1)))
t_critico = t.ppf(confianza, df=gl)
margen_error = t_critico * s

limite_inferior = diff_medias - margen_error
limite_superior = math.inf

print(f'Intervalo de confianza al 95% para μA - μB: [{limite_inferior:.2f}, {limite_superior:.2f}]')
print(f'Interpretación: con 95% de confianza, el programa de tutoría aumenta las notas al menos en {limite_inferior:.2f} puntos.\n')

# Conclusiones
print('--- CONCLUSION  ---')
print(f'-> El promedio del Grupo A (Tutoría) fue {media_a:.2f}, mientras que el del Grupo B (Control) fue {media_b:.2f}.')
print(f'-> La diferencia de medias es de {diff_medias:.2f} puntos.')
print(f'-> Según la prueba t: {conclusion_test}')
print(f'-> El intervalo de confianza indica que el efecto mínimo positivo de la tutoría es de {limite_inferior:.2f} puntos (con 95% de confianza).')
print('-> La evidencia estadística respalda que el programa de tutoría mejora significativamente el rendimiento académico.')
