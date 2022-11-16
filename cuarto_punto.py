"""
Observaciones:
1.  Cargar las bases de datos como Dataframe
2.  Dividir la base de datos en 2 partes: Train y test. Garantizar que la distribución de clases sea igual en el set de Train y Test.
3.  Seleccione un modelo de clasificación de cada uno de los tipos de modelos presentados a continuación: 

> a) Modelos semi-estadisticos (KNN, Naive Bayes, LDA)
> b) Modelos por umbral (Decisión trees, random forests, LogisticRegression)
> c) Modelos por identificación de frontera de decisión (SVM, Perceptron, Fuzzy learning, Ensemble)

4. Proponga un esquema de entrenamiento y validación de los modelos seleccionados.
   El entrenamiento deberá garantizar que tuvo en cuenta tecnicas de balance de clases (Se sugiere SMOTE o ADASYN). Así mismo, la etapa de validación deberá tener su respectiva matriz de confusión y el cálculo de un reporte de desempeño con métricas como: Sensibilidad, Accuracy, f1 score, entre otros.

5. Realice el entrenamiento y validación. Reporte los resultados.

6. Concluya cuál es el mejor clasificador y por qué. Así mismo, justifique la técnica utilizada para la selección de hiperparámetros en los clasificadores que aplique. 
"""
"""
===================================================
IMPORTACION LIBRERIAS
===================================================
"""
# Cargamos la informacion en el dataframe, esta data se encuentra delimitazda por comas, por defecto read_csv lo lee sin problemas
# Pero por buena practica defini el tipo de separacion, Tambien defini que mi columna INDEX fuera el Id del pasajero.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv(r'C:\Users\Sebastian\Documents\Python Scripts\PRUEBA TECNICA\Prueba tecnica\Data_titanic.csv',
                 sep=',', index_col='Passengerid')
# Informacion de las primeras 20 Filas.
print(df.head())
# Consulto las columnas disponibles

print('Columnas Iniciales')
print(df.columns)
# Columnas a eliminar del Dataframe
drop_list = [
    'zero',
    'zero.1',
    'zero.2',
    'zero.3',
    'zero.4',
    'zero.5',
    'zero.6',
    'zero.7',
    'zero.8',
    'zero.9',
    'zero.10',
    'zero.11',
    'zero.12',
    'zero.13',
    'zero.14',
    'zero.15',
    'zero.16',
    'zero.17',
    'zero.18'
]
df.drop(drop_list, axis=1, inplace=True)
print('Columnas Finales')
print(df.columns.values)
print(df.info())
# Validando datos faltantes en el DATASET
print(
    f"""
Informacion Faltante:
{df.isnull().sum()}
""")
# Revisando estadisticas del Dataset
print(df.describe())
# Eliminar la informacion de los datos faltantes
df.dropna(axis=0, how='any', inplace=True)
# Nuevamente Validando datos faltantes en el DATASET
print(
    f"""
Informacion Faltante:
{df.isnull().sum()}
""")

X = np.array(df.drop(['2urvived'], 1))
y = np.array(df['2urvived'])

# Informacion de Entranamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.33,
                                                    shuffle=True)
# Regresion Logistica


def modelo_regresion_logistica():
    regresion_logistica = LogisticRegression()
    regresion_logistica.fit(X_train, y_train)
    y_pred = regresion_logistica.predict(X_test)
    print('Precision Regresion Logistica: ',
          regresion_logistica.score(X_train, y_train))


def modelo_maquinas_soporte():
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print('Precision Maquinas de Soporte: ',
          svc.score(X_train, y_train))

def modelo_vecinos_cercanos():
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Precision Vecinos mas cercanos: ',
          knn.score(X_train, y_train))
    
    
modelo_regresion_logistica()
modelo_maquinas_soporte()
modelo_vecinos_cercanos()


