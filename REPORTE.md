# Práctica 3 · Resultados de clasificación

**Repositorio de GitHub:** https://github.com/arturozamorareyes-art/practica3

## Pregunta experimental

¿Cómo cambia la exactitud de seis clasificadores al pasar de flores medidas a datos con una perturbación compartida entre rasgos? La unidad de evaluación es una muestra clasificada correctamente. Exactitud = aciertos / muestras evaluadas.

## Datos y protocolo

Iris contiene 150 flores, 50 por especie (setosa, versicolor y virginica). Sus cuatro rasgos son longitud y ancho del sépalo y del pétalo en centímetros. El conjunto sintético contiene 240 observaciones, 80 por clase. Cada clase tiene un centro diferente y dispersión gaussiana de desviación 0.8. En 60 observaciones elegidas sin reemplazo se suma 1.8 a los dos primeros rasgos simultáneamente. Esto simula un sesgo compartido entre sensores y mantiene la etiqueta original.

Hold Out separa 105/45 flores y 168/72 observaciones sintéticas. K-Fold estratificado usa cinco pliegues externos. Cada muestra aparece una vez como prueba. Todos los modelos usan exactamente las mismas particiones. La estandarización estima media y desviación por rasgo en cada entrenamiento.

Para K-NN, SVM y MLP se comparó la configuración inicial con una búsqueda por rejilla. En cada entrenamiento externo, tres pliegues internos seleccionan la mayor exactitud media, con desempate por orden de la rejilla. Se reentrena la configuración elegida y se mide en la prueba externa. Así se obtiene una estimación del rendimiento de seleccionar parámetros sobre datos nuevos. Las opciones ganadoras de cada partición constan en el CSV. La base usa k=5, SVM con C=1 y MLP con una capa de 100 neuronas y Adam. La búsqueda considera k∈{5,3,9}, C∈{1,0.1,10}, y capas de 100 o 24 neuronas con Adam o L-BFGS.

## Exactitud observada (%)

La columna K-Fold es la media de los cinco porcentajes. Δ expresa el cambio en puntos porcentuales al ajustar parámetros. Un guion identifica modelos evaluados solamente con su configuración base.

| Datos | Modelo | Hold Out base | Ajustado | Δ | K-Fold base | Ajustado | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Iris | Distancia mínima | 82.22 | — | — | 86.67 | — | — |
| Iris | 1-NN | 93.33 | — | — | 93.33 | — | — |
| Iris | K-NN | 91.11 | 91.11 | +0.00 | 97.33 | 95.33 | -2.00 |
| Iris | Naïve Bayes | 91.11 | — | — | 94.67 | — | — |
| Iris | SVM | 93.33 | 93.33 | +0.00 | 96.00 | 96.00 | +0.00 |
| Iris | MLP | 88.89 | 91.11 | +2.22 | 95.33 | 94.67 | -0.67 |
| Sintético | Distancia mínima | 94.44 | — | — | 85.42 | — | — |
| Sintético | 1-NN | 87.50 | — | — | 82.92 | — | — |
| Sintético | K-NN | 91.67 | 91.67 | +0.00 | 84.58 | 86.25 | +1.67 |
| Sintético | Naïve Bayes | 94.44 | — | — | 85.00 | — | — |
| Sintético | SVM | 91.67 | 91.67 | +0.00 | 89.58 | 87.92 | -1.67 |
| Sintético | MLP | 90.28 | 90.28 | +0.00 | 88.33 | 88.33 | +0.00 |

## Interpretación

En Iris, K-NN base obtuvo 146 aciertos de 150 en las predicciones externas de K-Fold (97.33 %). Ajustar k costó tres aciertos. En el sintético se observó lo contrario: el ajuste de k añadió cuatro aciertos de 240. SVM base encabezó K-Fold en el sintético con 215 de 240 aciertos (89.58 %). Su ajuste perdió cuatro. La mejor configuración dentro de una rejilla puede cambiar con la muestra de entrenamiento.

En Hold Out sintético, distancia mínima y Bayes empataron con 68 de 72 aciertos. Su posición cambió en K-Fold. Con Iris, una sola flor cambia el resultado de Hold Out en 2.22 puntos. Las diferencias pequeñas deben interpretarse con esa resolución. La práctica utiliza una semilla y una realización de ruido. Para atribuir una pérdida al ruido haría falta repetir la evaluación con las mismas muestras antes y después de perturbarlas.

Se registraron 92 avisos de convergencia durante ajustes, incluidos los entrenamientos internos. El MLP puede alcanzar su límite de iteraciones antes de estabilizarse. Sus cifras describen el presupuesto predeterminado utilizado. El CSV distingue el tiempo de predicción del tiempo de ajuste, que incluye la búsqueda cuando se activa.

## Complejidad temporal

Sea n el número de muestras de entrenamiento, d los rasgos, c las clases, s los vectores de soporte y W el total de pesos y sesgos del MLP. La predicción se expresa por una muestra. El escalado añade O(nd) al entrenamiento y O(d) por predicción.

| Modelo | Entrenamiento | Predicción | Motivo |
|---|---|---|---|
| Distancia mínima | O(nd + cn + n log n) | O(cd) | Se ordenan etiquetas con unique, se recorre una máscara por clase y se suman los rasgos de cada grupo. |
| 1-NN | O(n log n) | O(nd) | fit codifica etiquetas con unique y conserva X por referencia. La consulta calcula todas las distancias y busca el mínimo. |
| K-NN | O(n log n) | O(nd + n log n + k + c) | La consulta ordena las distancias, toma k vecinos y cuenta votos. |
| Naïve Bayes gaussiano | O(nd) para c fijo | O(cd) | Estima media y varianza por rasgo y clase. Evalúa una puntuación por clase. |
| SVM radial | Aproximadamente entre cuadrático y cúbico en n en problemas densos, según datos, caché y convergencia; cada evaluación de kernel cuesta O(d) | O(sd + cs + c²) | SVC resuelve problemas por pares de clases. Reutiliza los kernels de los vectores de soporte y agrega votos. |
| MLP | O(E n W) para Adam con E épocas; para L-BFGS el coste dominante depende del número de evaluaciones completas de la función y del gradiente | O(W) | Las pasadas hacia delante y atrás recorren las conexiones de la red. |

Con g candidatos, la búsqueda ejecuta 3g entrenamientos internos y un ajuste final por partición externa. Los tamaños de muestra y los criterios de parada también influyen en el tiempo observado.

## Referencias

- [Iris, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)
- [Validación cruzada en scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Complejidad de SVM](https://scikit-learn.org/stable/modules/svm.html#complexity)
- [Perceptrón multicapa](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
