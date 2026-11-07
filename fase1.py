import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris


def datasets():
    rng = np.random.default_rng(42)
    centros = np.array([[0, 0, 1, 0], [2, 0, 0, 1], [0, 2, 2, 1]])
    clases = np.repeat(np.arange(3), 80)
    X = centros[clases] + rng.normal(0, 0.8, (240, 4))
    # Un desplazamiento común representa una perturbación de dos sensores.
    afectados = rng.choice(len(X), size=60, replace=False)
    X[afectados, :2] += 1.8
    return {"Iris": load_iris(return_X_y=True), "Sintético": (X, clases)}


class DistanciaMinima(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centros_ = np.array([X[y == clase].mean(axis=0) for clase in self.classes_])
        return self

    def predict(self, X):
        indices = [np.argmin(np.sum((self.centros_ - x) ** 2, axis=1)) for x in X]
        return self.classes_[indices]


class Vecinos(ClassifierMixin, BaseEstimator):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        if not 1 <= self.k <= len(X):
            raise ValueError("k debe estar entre 1 y el número de muestras.")
        self.X_ = X
        self.classes_, self.etiquetas_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        predicciones = []
        for x in X:
            distancias = np.sum((self.X_ - x) ** 2, axis=1)
            if self.k == 1:
                ganador = self.etiquetas_[np.argmin(distancias)]
            else:
                cercanos = np.argsort(distancias, kind="stable")[:self.k]
                votos = np.bincount(self.etiquetas_[cercanos])
                ganador = np.argmax(votos)
            predicciones.append(self.classes_[ganador])
        return np.array(predicciones)
