# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

El conjunto de datos "Jigsaw Unintended Bias in Toxicity Classification" comprende una compilación de comentarios extraídos de diversas plataformas en línea, cada uno acompañado de etiquetas cruciales para su clasificación.

Tamaño del conjunto de datos: Incluye una cantidad significativa de observaciones, lo que garantiza una representación amplia y variada de comentarios en línea.

Variables clave:

Comentario: Contiene el texto completo de cada comentario.
Toxicidad: Una etiqueta binaria que indica si el comentario es considerado tóxico o no.
Grado de severidad: Proporciona una clasificación sobre la intensidad del contenido tóxico.
Sesgo de identidad: Identifica cualquier sesgo relacionado con la identidad presente en el comentario, abordando cuestiones como el racismo o el sexismo.
Tipos de variables:

Los comentarios se presentan como cadenas de texto, mientras que las etiquetas de toxicidad, grado de severidad y sesgo de identidad son variables categóricas.
Integridad de los datos: Es fundamental examinar la presencia de valores faltantes para garantizar la coherencia y calidad del conjunto de datos.

Distribución de variables: Es esencial comprender la distribución de las etiquetas de toxicidad y los sesgos de identidad para abordar posibles desequilibrios de clase y sesgos inherentes

## Resumen de calidad de los datos

Valores faltantes:
Se realizó una exhaustiva revisión de la presencia de valores faltantes en todas las variables del conjunto de datos. Se determinó que el X% de las observaciones presentaban valores faltantes en la variable Y. Para manejar esta situación, se implementaron las siguientes estrategias:

Imputación de datos: Se aplicaron técnicas de imputación, como el uso de la media, la mediana o la moda, para rellenar los valores faltantes, teniendo en cuenta el tipo de variable y la distribución de los datos. Dado que se trata de datos de texto, se consideraron métodos específicos para el imputado de datos faltantes en este contexto, como la imputación basada en texto o el uso de embeddings.
En situaciones donde la imputación no era adecuada o no era posible, se optó por eliminar las observaciones con valores faltantes, preservando así la integridad del conjunto de datos.
Valores extremos:
Se llevaron a cabo análisis de valores extremos en las variables numéricas para detectar posibles anomalías o errores en los datos. Las acciones realizadas incluyeron:

Limpieza manual de errores tipográficos y corrección de datos erróneos, especialmente relevantes en el contexto del procesamiento de texto.
Eliminación de observaciones duplicadas para garantizar la unicidad de cada entrada en el conjunto de datos, evitando así la redundancia y la distorsión de los resultados de NLP.
Estas acciones se llevaron a cabo con el objetivo de asegurar la calidad y fiabilidad del conjunto de datos, aspectos fundamentales para el éxito de cualquier proyecto de procesamiento del lenguaje natural.

## Variable objetivo

Variable Objetivo
La variable objetivo en el conjunto de datos "Jigsaw Unintended Bias in Toxicity Classification" es la toxicidad. Esta variable es una etiqueta binaria que indica si un comentario es considerado tóxico o no. A continuación, se describe la distribución de esta variable y se presentan gráficos para comprender mejor su comportamiento.

Descripción de la Variable Objetivo
Toxicidad: Esta variable indica si un comentario es tóxico (1) o no tóxico (0).
Distribución de la Variable Objetivo
Para entender la distribución de la variable de toxicidad, se realiza un análisis de frecuencia de las clases tóxicas y no tóxicas.

Distribución de Clases:
Tóxico (1): Proporción de comentarios clasificados como tóxicos.
No tóxico (0): Proporción de comentarios clasificados como no tóxicos.
Gráficos
Para visualizar mejor la distribución de la variable objetivo, se presentan los siguientes gráficos:

Histograma de la Variable Objetivo: Este gráfico muestra la frecuencia de comentarios tóxicos y no tóxicos.
Gráfico de Barras: Un gráfico de barras que ilustra la proporción de comentarios tóxicos y no tóxicos en el conjunto de datos.
A continuación se muestran los gráficos:

python
Copiar código
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que 'df' es el DataFrame que contiene los datos y 'toxicity' es la variable objetivo
# Histograma de la variable objetivo
plt.figure(figsize=(8, 6))
sns.countplot(x='toxicity', data=df)
plt.title('Distribución de la Variable Objetivo (Toxicidad)')
plt.xlabel('Toxicidad')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras de la distribución de la variable objetivo
plt.figure(figsize=(8, 6))
df['toxicity'].value_counts(normalize=True).plot(kind='bar', color=['blue', 'orange'])
plt.title('Proporción de Comentarios Tóxicos y No Tóxicos')
plt.xlabel('Toxicidad')
plt.ylabel('Proporción')
plt.show()

Estos gráficos permiten visualizar claramente la distribución de la variable objetivo y entender mejor el comportamiento de los datos de toxicidad en el conjunto de datos.

## Variables individuales

Variables Individuales
En esta sección se presenta un análisis detallado de cada variable individual en el conjunto de datos. Para cada variable, se incluyen estadísticas descriptivas, gráficos de distribución y, si es relevante, gráficos que muestran la relación con la variable objetivo (toxicidad). También se describen posibles transformaciones que podrían aplicarse a las variables.

1. Variables Decimales
Descripción y Estadísticas Descriptivas

Se proporcionan estadísticas como la media, la mediana, la desviación estándar, los valores mínimos y máximos.
python
Copiar código
# Ejemplo de cálculo de estadísticas descriptivas para variables decimales
decimal_columns = df.select_dtypes(include=['float64'])
decimal_descriptive_stats = decimal_columns.describe()

import ace_tools as tools; tools.display_dataframe_to_user(name="Estadísticas Descriptivas de Variables Decimales", dataframe=decimal_descriptive_stats)
Gráficos de Distribución

Histogramas y gráficos de densidad para visualizar la distribución de cada variable decimal.
python
Copiar código
# Ejemplo de gráfico de distribución para una variable decimal
for column in decimal_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()
Relación con la Variable Objetivo

Gráficos de dispersión y boxplots que muestran la relación entre cada variable decimal y la toxicidad.
python
Copiar código
# Ejemplo de gráfico de relación entre una variable decimal y la toxicidad
for column in decimal_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='toxicity', y=column, data=df)
    plt.title(f'Relación entre {column} y Toxicidad')
    plt.xlabel('Toxicidad')
    plt.ylabel(column)
    plt.show()
Posibles Transformaciones

Normalización o estandarización para variables con escalas muy diferentes.
Transformaciones logarítmicas para variables con distribución sesgada.
2. Variables Enteras
Descripción y Estadísticas Descriptivas

Se proporcionan estadísticas como la media, la mediana, la desviación estándar, los valores mínimos y máximos.
python
Copiar código
# Ejemplo de cálculo de estadísticas descriptivas para variables enteras
integer_columns = df.select_dtypes(include=['int64'])
integer_descriptive_stats = integer_columns.describe()

tools.display_dataframe_to_user(name="Estadísticas Descriptivas de Variables Enteras", dataframe=integer_descriptive_stats)
Gráficos de Distribución

Histogramas para visualizar la distribución de cada variable entera.
python
Copiar código
# Ejemplo de gráfico de distribución para una variable entera
for column in integer_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()
Relación con la Variable Objetivo

Gráficos de dispersión y boxplots que muestran la relación entre cada variable entera y la toxicidad.
python
Copiar código
# Ejemplo de gráfico de relación entre una variable entera y la toxicidad
for column in integer_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='toxicity', y=column, data=df)
    plt.title(f'Relación entre {column} y Toxicidad')
    plt.xlabel('Toxicidad')
    plt.ylabel(column)
    plt.show()
Posibles Transformaciones

Binning (agrupamiento en categorías) para variables con muchos valores únicos.
Escalado (min-max scaling) para ajustar las variables a un rango específico.
3. Variables UUID
Descripción y Estadísticas Descriptivas

Generalmente, las variables UUID se utilizan como identificadores únicos y no se analizan estadísticamente de la misma manera que otras variables numéricas o categóricas.
Posibles Transformaciones

En general, los UUID no requieren transformaciones a menos que se utilicen como índices o claves.
4. Otras Variables
Descripción y Estadísticas Descriptivas

Se proporcionan estadísticas descriptivas específicas dependiendo del tipo de variable (por ejemplo, categóricas, booleanas, etc.).
Gráficos de Distribución y Relación con la Variable Objetivo

Se utilizan gráficos específicos según el tipo de variable (gráficos de barras para variables categóricas, gráficos de dispersión para variables continuas, etc.).
python
Copiar código
# Ejemplo de gráfico de barras para una variable categórica
categorical_columns = df.select_dtypes(include=['object'])
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()
Posibles Transformaciones

Codificación de variables categóricas (One-Hot Encoding, Label Encoding).
Conversión de variables booleanas a enteras (0 y 1).
Este análisis detallado y los gráficos asociados proporcionan una visión clara de cada variable en el conjunto de datos, lo que es fundamental para el desarrollo y evaluación de modelos de machine learning en proyectos de NLP.

## Ranking de variables

Ranking de las Variables Más Importantes
En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo (toxicidad). Se emplean técnicas como la correlación, el análisis de componentes principales (PCA) y la importancia de las variables en modelos de aprendizaje automático para identificar y clasificar estas variables.

1. Análisis de Correlación
Descripción:

Se calcula la correlación entre cada variable y la variable objetivo para identificar qué variables tienen una relación significativa con la toxicidad.
Método:

python
Copiar código
# Cálculo de la correlación con la variable objetivo
correlation_matrix = df.corr()
correlation_with_target = correlation_matrix['toxicity'].sort_values(ascending=False)

tools.display_dataframe_to_user(name="Correlación con la Variable Objetivo", dataframe=correlation_with_target)
2. Análisis de Componentes Principales (PCA)
Descripción:

PCA se utiliza para reducir la dimensionalidad del conjunto de datos y para identificar las variables que más contribuyen a la varianza explicada en los datos.
Método:

python
Copiar código
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Estandarización de los datos
features = df.drop(columns=['toxicity'])
features_scaled = StandardScaler().fit_transform(features)

# Aplicación de PCA
pca = PCA(n_components=10)  # Suponiendo que queremos los 10 componentes principales
principal_components = pca.fit_transform(features_scaled)

# Cálculo de la importancia de las variables
pca_importance = pd.DataFrame(pca.components_, columns=features.columns)
pca_importance = pca_importance.abs().sum().sort_values(ascending=False)

tools.display_dataframe_to_user(name="Importancia de las Variables según PCA", dataframe=pca_importance)
3. Importancia de las Variables en Modelos de Aprendizaje Automático
Descripción:

Se entrena un modelo de aprendizaje automático (por ejemplo, un modelo de bosque aleatorio) y se extrae la importancia de las variables a partir del modelo entrenado.
Método:

python
Copiar código
from sklearn.ensemble import RandomForestClassifier

# Preparación de los datos
X = df.drop(columns=['toxicity'])
y = df['toxicity']

# Entrenamiento del modelo de bosque aleatorio
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Cálculo de la importancia de las variables
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

tools.display_dataframe_to_user(name="Importancia de las Variables en Bosque Aleatorio", dataframe=feature_importances)
Ranking de las Variables Más Importantes
A continuación, se muestra un ranking consolidado de las variables más importantes basado en los métodos de análisis de correlación, PCA y modelos de aprendizaje automático. Estas técnicas ayudan a identificar las variables clave que influyen en la predicción de la toxicidad.

Ranking Consolidado:

Variable A - Alta correlación, alta importancia en PCA y modelos.
Variable B - Alta importancia en modelos de bosque aleatorio.
Variable C - Alta contribución en PCA.
Variable D - Significativa correlación con la variable objetivo.
Variable E - Consistente importancia a través de múltiples técnicas.
Este análisis proporciona una visión clara de las variables más relevantes para predecir la toxicidad, lo que es fundamental para construir y mejorar modelos de machine learning en proyectos de NLP.
