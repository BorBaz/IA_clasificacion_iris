import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Se carga la info sobre las flores
iris = load_iris()

# Se dividen los datos para datos de entrenamiento y datos de prueba
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# El modelo de clasificación se crea utilizando una red neuronal artificial (ANN) con capas densas y funciones de activación. Durante el entrenamiento del modelo, se ajustan los pesos de las capas para minimizar la función de pérdida, y se utiliza un algoritmo de optimización (en este caso, Adam) para actualizar los pesos en cada iteración del entrenamiento.
# Una vez que el modelo está entrenado, se utiliza para hacer predicciones en nuevos datos. En el caso de nuestro modelo de clasificación de especies de flores iris, dada una medición de longitud y ancho de los sépalos y pétalos, el modelo predice a qué especie de iris pertenece la flor.
# En resumen, un modelo de clasificación es un modelo de aprendizaje automático que se entrena para asignar entradas de datos a categorías predefinidas, y se utiliza para hacer predicciones en nuevos datos.
# Creamos el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

#
# La compilación del modelo se realiza utilizando el método compile() de Keras, que se utiliza para configurar el proceso de entrenamiento del modelo. En la función compile(), especificamos la función de pérdida, el optimizador y la métrica que se utilizarán para evaluar el rendimiento del modelo durante el entrenamiento.
# La función de pérdida es una medida de qué tan bien el modelo está haciendo predicciones en los datos de entrenamiento. El optimizador es un algoritmo que se utiliza para ajustar los pesos del modelo durante el entrenamiento. La métrica es una medida adicional que se utiliza para evaluar el rendimiento del modelo durante el entrenamiento.
# En nuestro ejemplo de código, especificamos la función de pérdida categorical_crossentropy, que es una función comúnmente utilizada en problemas de clasificación con múltiples categorías. También utilizamos el optimizador Adam, que es un algoritmo de optimización popular utilizado en el aprendizaje profundo. Por último, especificamos la métrica accuracy, que es una medida comúnmente utilizada para evaluar el rendimiento de los modelos de clasificación.
# Por lo tanto, compilar el modelo es un paso necesario en el proceso de entrenamiento del modelo, ya que nos permite configurar el proceso de entrenamiento con la función de pérdida, el optimizador y la métrica adecuados para nuestro problema de clasificación de especies de iris.
# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Entrenar el modelo
model.fit(x_train, y_train, epochs=60)


# Evaluar el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Accuracy:', test_acc)