# -*- coding: utf-8 -*-
"""
Prueba técnica Científico de datos - Skill: Programación (10 pts)
Un problema típico para empresas de datos es tener grandes cantidades de información desordenada.
Una de las tareas iniciales de cada proyecto es organizar la data.

En este ejemplo, contamos con un vector con números desordenados los cuales deben ser ordenados.
Para hacerlo, su supervisor le indica que debe realizar una función que tenga como entrada un array de n cantidad de elementos de tipo integer y ordenarlo mediante la técnica de insertion.
Esta técnica puede observarse en la Figura 1.

Criterios de evaluación

El candidato demuestra un adecuado uso de las estructuras cíclicas (for, while) (2 pts)
El candidato utiliza de forma adecuada estructuras condicionales (2 pts)
El candidato demuestra una adecuada lógica de programación (6 pts)
"""


def ordanamiento_arreglo(arreglo):
    longitud = len(arreglo)
    for i in range(longitud):
        for actual in range(longitud - 1):
            siguiente = actual + 1
            if arreglo[actual] > arreglo[siguiente]:
                arreglo[siguiente], arreglo[actual] = arreglo[actual], arreglo[siguiente]
    return arreglo


def run():
    # Vector de prueba prueba_data
    prueba_data = [20, 3, 5, 90, 120, 50, 25, 33, 31]
    print(
    f"""
    =========================================================
    Ordenamiento de informacion
    ARREGLO INICIAL: {prueba_data}
    =========================================================
    """
    )
    data_resultado = ordanamiento_arreglo(prueba_data)
    print(
    f"""
    =========================================================
    Ordenamiento de informacion
    ARREGLO FINAL: {data_resultado}
    =========================================================
    """
    )
    # Tambien podiamos utilizar el metodo sort


if __name__ == '__main__':
    run()
