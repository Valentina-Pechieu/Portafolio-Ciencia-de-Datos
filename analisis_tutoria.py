import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t
import math

grupo_a=np.array([85,90,78,88,92,80,86,89,84,87,91,82,83,85,88])
grupo_b=np.array([70,72,75,78,80,68,74,76,79,77,73,71,75,78,80])

#1. Diseño del Experimento 
#Para disminuir el sesgo, se podria aumentar la cantidad de la muestra al menos a 30.
#Obtener los siguientes datos: la cantidad de asistencias, el tiempo que llevan en las tutorias, el rango de edad de los estudiantes, la cantidad de cada genero por cada grupo.  

#2. Cálculo de Estadísticas Descriptivas
media_a=np.mean(grupo_a)
desv_a=np.std(grupo_a)
print(f'La Media del Grupo A es: {media_a:.2f}')
print(f'La Desviacion Estandar del Grupo A es: {desv_a:.2f}\n')

media_b=np.mean(grupo_b)
desv_b=np.std(grupo_b)
print(f'La Media del Grupo B es: {media_b:.2f}')
print(f'La Desviacion Estandar del Grupo B es: {desv_b:.2f}\n')

#Boxplot
plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
sns.boxplot(grupo_a, color='skyblue')
plt.title('Boxplot del Grupo A', y=1.02)
plt.grid(True, axis='y', linestyle='--', alpha=0.4 )

plt.subplot(1,2,2)
sns.boxplot(grupo_b, color='skyblue')
plt.title('Boxplot del Grupo B', y=1.02)
plt.grid(True, axis='y', linestyle='--', alpha=0.4 )
plt.show()

#Histograma
plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
sns.histplot(grupo_a, bins=5, kde=True, color='skyblue', edgecolor='black')
plt.title('Histograma del Grupo A', y=1.02)
plt.xlabel('Notas de examen')
plt.ylabel('Frecuencia')
plt.tight_layout()

plt.subplot(1,2,2)
sns.histplot(grupo_b, bins=5, kde=True, color='skyblue', edgecolor='black')
plt.title('Histograma del Grupo B', y=1.02)
plt.xlabel('Notas de examen')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

#3. Prueba de Hipótesis
#H0: μ(A)​=μ(B​)
#H1: μ(A​)>μ(B)​

#Prueba t 
alfa=0.05

t_stat, p_valor=stats.ttest_ind(grupo_a, grupo_b, equal_var=False)

if t_stat > 0:
    p_valor_uni=p_valor/2
else:
    p_valor_uni=1-(p_valor / 2)

print(f't-Estadistico: {t_stat:.4f}')
print(f'p-valor: {p_valor_uni:.4f}')

if p_valor_uni<alfa:
    print('\nSe rechaza H0. La media de A es significativamente diferente de la media de B')
else:
    print('\nNo se rechaza H0. No hay envidencia para concluir que los resultados son significativos')

#4. Intervalo de Confianza
n=15
confianza=0.95
s_a=np.var(grupo_a, ddof=1)
s_b=np.var(grupo_b, ddof=1)

x=media_a-media_b
s=np.sqrt(s_a/n+s_b/n)

grados_libertad=(s_a/n+s_b/n)**2/((s_a**2)/(n**2*(n-1))+(s_b**2)/(n**2*(n-1)))

t_critico=t.ppf(1-(1-confianza),df=grados_libertad)
margen_error=t_critico*s

limite_inferior=x-margen_error
limite_superior=math.inf

print(f'\nEl intervalo de confianza 95%: [{limite_inferior:.2f}, {limite_superior}]')
print(f'El programa de tutoria aumenta las notas de los estudiantes en mas de {limite_inferior:.2f} puntos')