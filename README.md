# MachineLearning

Implementación de Máquina de Vectores Soporte para Regresión Modo linear: linearsvr.py Modo completo: completesvr.py

Máquina de vectores sopote para regresión es un sistema matemático de minimización con restricciones, se resuelve utilizando multiplicadores de Lagrange.
Su objetivo es encontrar la recta que mejor se ajuste a los datos que se ajuste con un menor error. Al trabajar únicamente con rectas es necesario transformar el plano de forma que los datos en este se ajusten a una recta de forma que el algoritmo se ajuste a cualquier tipo de función, para esto se utilizan los llamados kernels.

# Perceptrón Multicapa mlp.py

El Perceptrón multicapa es una red neuronal con una capa de entrenada, otra de salida y múltiples capas ocultas cuyos pesos se ajustan por medio del algoritmo Gradiente Descendente. Los valores "fluyen" por la red hasta alcanzar la capa de salida en donde se calcula dicho gradiente, una vez calculado este se retropropaga hacia la entrada, corrigiendo los pesos de las neuronas en el proceso, de forma que en cada iteración el error disminuya. Se debe seleccionar cuidadosamente el escalón de aprendizaje, ya que valores elevados pueden resultar que el algoritmo nunca converja hacia una solución óptima. 


# Máquina de aprendizaje extremo elm.py

Máquina de aprendizaje extremo consiste en una versión del algoritmo MLP (Perceptrón multicapa) en donde únicamente existe un capa oculta además de la de entrada y salida, los valores de cada neurona de esta capa se establece una sola vez, sin existir iteraciones como en MLP, estos pesos iniciales se establecen según los datos de entrenamiento: Se multiplica la matriz speudo inversa de la capa de entrada por los valores objetivo (y), de esta forma se obtiene un algoritmo extremadamente eficiente y rápido.


# Bosque aleatorio ranforest.py (Implementación con errores)

Desde la experiencia este se trata del algoritmo más eficiente ya que implícitamente realiza una selección de características a la hora de construir cada árbol. Este algoritmo consiste en una agrupación de árboles de decisión, a cada uno de ellos se les asigna conjuntos de datos diferentes, la predicción final consiste en un promedio de cada uno, de forma que los errores se minimicen.
