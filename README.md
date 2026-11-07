# Práctica 3 · Comparación de seis clasificadores

El reporte con resultados y complejidad se encuentra en [REPORTE.md](REPORTE.md). Las tres fases comparten la misma interfaz de entrenamiento y predicción.

## Ejecutar desde Ubuntu

Desde la carpeta que contiene este README, usa Python 3.11 o 3.12 y ejecuta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python fase3.py
```

El último comando ejecuta las tres fases y actualiza resultados/experimentos.csv. El benchmark viene incluido en scikit-learn. Las semillas fijas reproducen los datos y las particiones. Los tiempos dependen del equipo.

## Cómo leer cada bloque

| Fase y bloque | Qué hace y por qué |
|---|---|
| 1 · datasets | Devuelve dos pares (X, y). Cada fila de X contiene cuatro mediciones y y indica la clase. En el sintético, una misma perturbación desplaza dos variables a la vez. |
| 1 · DistanciaMinima.fit | Calcula el promedio de cada rasgo dentro de cada clase para formar su centroide. |
| 1 · DistanciaMinima.predict | Elige el centroide con menor distancia euclidiana al cuadrado. Omitir la raíz conserva el orden de las distancias. |
| 1 · Vecinos.fit | Conserva las muestras de entrenamiento y codifica sus etiquetas como índices para contar votos. |
| 1 · Vecinos.predict | Con un vecino basta buscar el mínimo. Con varios, ordena por distancia y cuenta los votos de los primeros k. A igual distancia conserva el orden de entrenamiento. Un empate de votos favorece la etiqueta menor. |
| 2 · modelos | Construye seis clasificadores. GaussianNB estima medias y varianzas por clase. SVC usa un kernel radial. MLP aprende pesos entre capas. |
| 2 · REJILLAS | Define opciones para k, la penalización C de SVM, y la anchura y el optimizador de MLP. El prefijo modelo__ dirige el parámetro al clasificador dentro de la secuencia de procesamiento. |
| 3 · particiones | Produce índices comunes a todos los modelos. Estratificar conserva aproximadamente las proporciones de las clases. |
| 3 · evaluar | Ajusta escala y modelo con el entrenamiento disponible. Cuando hay rejilla, reserva pliegues internos para elegir parámetros. Evalúa una vez sobre la prueba externa. Registra aciertos, tiempos y avisos. |
| 3 · ejecutar | Recorre datasets, particiones, modelos y variantes. Guarda una fila por evaluación externa. |

## Reparto sugerido

| Integrante | Fase 1 | Fase 2 | Fase 3 |
|---|---|---|---|
| 1 | Distancia mínima y benchmark | Naïve Bayes y rejilla de vecinos | Analizar distancia mínima y Bayes |
| 2 | 1-NN y diseño del ruido | SVM y penalización | Analizar 1-NN y SVM |
| 3 | K-NN y representación de etiquetas | MLP y configuraciones | Analizar K-NN y MLP |
| 4 | Registrar metodología | Documentar configuraciones | Integrar reporte, repositorio y conversaciones |

## Entrega

Sube el contenido de esta carpeta a GitHub conservando su estructura. Sustituye el campo de repositorio del reporte por la URL que genere GitHub. Adjunta los historiales completos del chat inicial y de este chat a la entrega del curso. El [chat inicial](https://chatgpt.com/share/6aef15cb-0a08-83e8-99e5-255d304243f8) contiene la primera explicación. El diseño, código, ejecución y redacción de este proyecto se realizaron con ChatGPT.
