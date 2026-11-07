from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from fase1 import DistanciaMinima, Vecinos


def modelos():
    return {
        "Distancia mínima": DistanciaMinima(),
        "1-NN": Vecinos(1),
        "K-NN": Vecinos(),
        "Naïve Bayes": GaussianNB(),
        "SVM": SVC(),
        "MLP": MLPClassifier(random_state=42),
    }


# Cada lista incluye el valor inicial como referencia.
REJILLAS = {
    "K-NN": {"modelo__k": [5, 3, 9]},
    "SVM": {"modelo__C": [1, 0.1, 10]},
    "MLP": {"modelo__hidden_layer_sizes": [(100,), (24,)], "modelo__solver": ["adam", "lbfgs"]},
}
