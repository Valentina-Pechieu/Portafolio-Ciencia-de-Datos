#====================================================================
# Análisis de migraciones con PySpark y MLlib
#====================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, desc, count
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creación de la SparkSession
spark = (
    SparkSession.builder
    .appName('migraciones')
    .getOrCreate()
)
print('SparkSession creado con éxito')

# Carga del CSV en DataFrame
df = (
    spark.read
    .option('header', True)
    .option('inferSchema', True)
    .option('encoding', 'UTF-8')
    .csv('migraciones.csv')
)

# Conversión a RDD
rdd = df.rdd

print('Primeras 2 filas:')
df.show(2)

print('Esquema del DataFrame:')
df.printSchema()

print('Estadísticas descriptivas:')
df.describe().show(truncate=False)

# Procesamiento con RDDs
rdd_filtrado = rdd.filter(lambda fila: fila['Año'] >= 2017)
rdd_pares_origen = rdd_filtrado.map(lambda fila: (fila['Origen'], 1))
rdd_razones = rdd.flatMap(lambda fila: [(fila['Razón'], 1)])

total = rdd.count()
total_filtrado = rdd_filtrado.count()
print(f'Total de registros: {total}')
print(f'Registros filtrados (Año >= 2017): {total_filtrado}\n')

print(f'Primeras 3 filas del RDD filtrado:\n {rdd_filtrado.take(3)}')

conteo_por_origen = rdd_pares_origen.reduceByKey(lambda a, b: a + b).collect()
conteo_por_razon = rdd_razones.reduceByKey(lambda a, b: a + b).collect()

print(f'\nConteo por Origen: {conteo_por_origen}')
print(f'Conteo por Razón: {conteo_por_razon}')

# Procesamiento con DataFrames
df_razon = df.filter(df.Razón == 'Económica')
print(f'Total de registros con Razón Económica: {df_razon.count()}')

origen = (
    df_razon.groupBy('Origen')
    .agg(count('*').alias('total'))
    .orderBy(desc('total'))
)

destino = (
    df_razon.groupBy('Destino')
    .agg(count('*').alias('total'))
    .orderBy(desc('total'))
)

print('\nOrígenes por razones económicas')
origen.show()
print('\nDestinos por razones económicas')
destino.show()

# Exportación a formato Parquet
origen.write.mode('overwrite').parquet('origenes_razon_economico.parquet')
destino.write.mode('overwrite').parquet('destinos_razon_economico.parquet')
print('DataFrames guardados exitosamente en formato Parquet')

# Consultas con Spark SQL
df_sql = df.withColumn(
    'Region_Origen',
    when(col('Origen').isin('México','Venezuela','Argentina','Colombia','EEUU'), 'América')
    .when(col('Origen').isin('Alemania','España'), 'Europa')
    .when(col('Origen').isin('India','Siria'), 'Asia')
    .otherwise('Otra')
)

df_sql.createOrReplaceTempView('migraciones')

origen_sql = spark.sql('''
          SELECT `Origen`, COUNT(*) AS total
          FROM migraciones
          GROUP BY `Origen`
          ORDER BY total DESC
''')
print('\nPaíses de origen')
origen_sql.show()

destinos_sql = spark.sql('''
          SELECT `Destino`, COUNT(*) AS total
          FROM migraciones
          GROUP BY `Destino`
          ORDER BY total DESC
''')
print('\nPaíses de destino')
destinos_sql.show()

razones_sql = spark.sql('''
          SELECT `Region_Origen` AS Region, `Razón` AS Razon, COUNT(*) AS total
          FROM migraciones
          GROUP BY `Region_Origen`, `Razón`
          ORDER BY Region ASC, total DESC
''')
print('\nRazones de migración por región')
razones_sql.show()

# MLlib: Regresión Logística
num_cols = [
    'PIB_Origen', 'PIB_Destino',
    'Tasa_Desempleo_Origen', 'Tasa_Desempleo_Destino',
    'Nivel_Educativo_Origen', 'Nivel_Educativo_Destino',
    'Población_Origen', 'Población_Destino',
    'Año'
]
cat_cols = ['Origen', 'Destino']

# Creación de columna label binaria: Económica = 1, otra = 0
df = df.withColumn('label',
                  when(col('Razón') == 'Económica',1)
                  .otherwise(0))

# Indexación de variables categóricas
index = [StringIndexer(inputCol=c, outputCol=f'{c}_idx', handleInvalid='keep') for c in cat_cols]

# Codificación one-hot
ohe = OneHotEncoder(inputCols=[f'{c}_idx' for c in cat_cols],
                    outputCols=[f'{c}_oh' for c in cat_cols])

# Ensamblado de variables
assembler = VectorAssembler(
    inputCols=num_cols + [f'{c}_oh' for c in cat_cols],
    outputCol='caracteristicas',
    handleInvalid='keep'
)

# Escalado de características
scaler = StandardScaler(
    inputCol='caracteristicas',
    outputCol='caracteristicas_scaled',
    withMean=False,
    withStd=True)

# Modelo de regresión logística
lr = LogisticRegression(featuresCol='caracteristicas_scaled', labelCol='label')

# Pipeline de procesamiento completo
pipeline = Pipeline(stages=index + [ohe, assembler, scaler, lr])

# División en entrenamiento y prueba
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Entrenamiento del modelo
model = pipeline.fit(train)

# Predicciones sobre el conjunto de prueba
predict = model.transform(test)

print('\nPredicciones (label vs prediction):')
predict.select('label','prediction').show()

# Evaluación del modelo
evaluator_acc = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
accuracy = evaluator_acc.evaluate(predict)
print(f'Accuracy:  {accuracy:.3f}')
