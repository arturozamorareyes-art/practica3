import csv
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fase1 import datasets
from fase2 import REJILLAS, modelos

SALIDA = Path(__file__).resolve().parent / "resultados"


def particiones(y):
    indices = np.arange(len(y))
    entrenamiento, prueba = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)
    yield "Hold Out", 1, entrenamiento, prueba
    kfold = StratifiedKFold(5, shuffle=True, random_state=42)
    for numero, (entrenamiento, prueba) in enumerate(kfold.split(indices, y), 1):
        yield "K-Fold", numero, entrenamiento, prueba


def evaluar(modelo, rejilla, X, y, entrenamiento, prueba):
    estimador = Pipeline([("escala", StandardScaler()), ("modelo", modelo)])
    if rejilla:
        interna = StratifiedKFold(3, shuffle=True, random_state=7)
        estimador = GridSearchCV(estimador, rejilla, cv=interna, scoring="accuracy", error_score="raise")
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = perf_counter()
        estimador.fit(X[entrenamiento], y[entrenamiento])
        entrenado = perf_counter()
        pred = estimador.predict(X[prueba])
        fin = perf_counter()
    return dict(aciertos=int(np.sum(pred == y[prueba])), total=len(prueba),
                ajuste_ms=round(1000 * (entrenado - inicio), 3),
                prediccion_ms=round(1000 * (fin - entrenado), 3),
                parametros=json.dumps(getattr(estimador, "best_params_", {})),
                avisos_convergencia=sum(issubclass(a.category, ConvergenceWarning) for a in avisos))


def ejecutar():
    SALIDA.mkdir(exist_ok=True)
    filas = []
    for datos, (X, y) in datasets().items():
        for metodo, pliegue, entrenamiento, prueba in particiones(y):
            for nombre, modelo in modelos().items():
                variantes = {"Base": None}
                if nombre in REJILLAS:
                    variantes["Ajustado"] = REJILLAS[nombre]
                for variante, rejilla in variantes.items():
                    resultado = evaluar(modelo, rejilla, X, y, entrenamiento, prueba)
                    filas.append(dict(datos=datos, metodo=metodo, pliegue=pliegue,
                                      modelo=nombre, variante=variante, **resultado))
    with (SALIDA / "experimentos.csv").open("w", newline="", encoding="utf-8") as archivo:
        tabla = csv.DictWriter(archivo, fieldnames=filas[0].keys())
        tabla.writeheader()
        tabla.writerows(filas)


if __name__ == "__main__":
    ejecutar()
