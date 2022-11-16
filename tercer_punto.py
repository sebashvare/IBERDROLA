"""
Tercer punto
Instrucciones:  
1. Cargar la base de datos como Dataframe
2. Realizar un análisis exploratorio de la base de datos intentando identificar variables linealmente separables así como cuáles de las variables numéricas están correlacionadas con la columna G3. (Generar un informe con gráficas y principales KPI identificados)
3. Transforme las variables categóricas a variables numéricas
4. Normalice los datos entre -1 y 1
5. Separe los datos en 3 conjuntos, Train, Test y Validation
6. Aplique algún método de selección de características (PCA, KBest, entre otros) con el conjunto de validación y justifique porque utilizó ese método
7. Prediga la variable G3 mediante un modelo de regresión entrteenado con todas las caracrísticas (excepto G1,G2 y G3)
8. Prediga la variable G3 mediante un modelo de regresión entrenado con las características óptimas. 
9. Concluya según sus hallazgos.
"""

"""
===============================
Librerias 
===============================
"""

# 1. Cargar la base de datos como Dataframe


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model # Modelo Regresion Lienal
from sklearn.tree import DecisionTreeRegressor # Modelo Arboles De Decision
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
df = pd.read_csv(
    r'C:\Users\Sebastian\Documents\Python Scripts\PRUEBA TECNICA\Prueba tecnica\student-mat.csv', sep=';')
# 1.1 Observemos las primeras 10 filas de informaciion.add()
df.head(10)

# 2. Realizar un análisis exploratorio de la base de datos intentando identificar variables linealmente separables así como 
# cuáles de las variables numéricas están correlacionadas con la columna G3. 
# (Generar un informe con gráficas y principales KPI identificados)
# Variables numericcas las encontramos con Describe
df.describe()
# Histogramas: Un histograma es un gráfico de barras que agrupa números en una serie de intervalos, especialmente cuando hay una variable infinita,
# como los pesos y las medidas.
sns.distplot(df['G3'])
plt.show()
variable_correlacion = df.corr()
# Mapa de calor, utiliza colores para comparar y contrastar números en un conjunto de datos.
sns.heatmap(variable_correlacion)
plt.show()

# Transforme las variables categóricas a variables numéricas
# Obtengo lista del nombre de todas las columnas
lista_columnas = df.columns.tolist()

"""
0 school
1 sex
2 age
3 address
4 famsize
5 Pstatus
6 Medu
7 Fedu
8 Mjob
9 Fjob
10 reason
11 guardian
12 traveltime
13 studytime
14 failures
15 schoolsup
16 famsup
17 paid
18 activities
19 nursery
20 higher
21 internet
22 romantic
23 famrel
24 freetime
25 goout
26 Dalc
27 Walc
28 health
29 absences
30 G1
31 G2
32 G3
"""

# 3. Transforme las variables categóricas a variables numéricas
# 4. Normalice los datos entre -1 y 1

# Esta variable va a tener toda la informacion del dataframe inicial pero sin la columna G3 que es la que vamos a predecir.
X = df.drop('G3', axis=1)
Y = df['G3']
# Con el metodo get_dummies() transfromo variables categoricas a variables numericas, tambien podia pasarle como parametro el listado de
# Columnnas a las que se le necesitaba realizar el proceso.
X = pd.get_dummies(X)

# Informacion para Entrenamiento y test, por convencion decclaramos XY Train, XY Test
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=.20, random_state=40)

"""
================================================================================
 Modelos a EJECUTAR:
 Regresion LINEAL
 Arboles de Decision
================================================================================
"""    

def modelo_regresion_lineal():
    regresion_lineal = linear_model.LinearRegression()
    regresion_lineal.fit(X_TRAIN, Y_TRAIN)  # Ajustamos le Modelo.
    prediccion = regresion_lineal.predict(X_TEST)
    error = Y_TEST - prediccion
    plt.scatter(Y_TEST, error, color='red')
    plt.show()
    desv_stand = np.sqrt(mean_squared_error(Y_TEST, prediccion))
    varianza = r2_score(Y_TEST, prediccion)
    print(
        f"""
    Modelo: Regresion Lineal
    Desviacion Estandar = {desv_stand}
    % Varianza = {varianza}
    """
    )


def modelo_arboles_decision():
    arboles_decision = DecisionTreeRegressor(max_features='auto')
    arboles_decision.fit(X_TRAIN, Y_TRAIN)
    prediccion = arboles_decision.predict(X_TEST)
    error = Y_TEST-prediccion
    plt.scatter(Y_TEST, error, color="red")
    plt.show()
    desv_stand = np.sqrt(mean_squared_error(Y_TEST, prediccion))
    varianza = r2_score(Y_TEST, prediccion)
    print(
        f"""
    Modelo: Regresion Lineal
    Desviacion Estandar = {desv_stand}
    % Varianza = {varianza}
    """
    )


print(
"""
================================================================================
 resultados modelo_regresion_lineal, Arboles de Decision
================================================================================
"""
)


modelo_regresion_lineal()
modelo_arboles_decision()
