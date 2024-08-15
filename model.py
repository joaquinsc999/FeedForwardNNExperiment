import tensorflow as tf
import numpy as np

class DenseLayer:
    """
    Clase que representa una capa densa (fully connected) en una red neuronal.

    Atributos:
    - num_neurons: Número de neuronas en la capa.
    - activation: Función de activación a utilizar (ej. tf.nn.relu).
    - weights: Tensor que representa los pesos de la capa.
    - biases: Tensor que representa los sesgos de la capa.
    - batch_size: Tamaño del batch para el entrenamiento.
    - prev_num_neurons: Número de neuronas en la capa anterior.
    """

    def __init__(self, num_neurons=1, activation=None, batch_size=1, prev_num_neurons=1):
        """
        Constructor de la clase DenseLayer.

        Parámetros:
        - num_neurons: Número de neuronas en la capa.
        - activation: Función de activación a aplicar.
        - batch_size: Tamaño del lote (batch size).
        - prev_num_neurons: Número de neuronas en la capa anterior.
        """
        self.num_neurons = num_neurons
        self.activation = activation
        # Inicialización de pesos con distribución normal, escalada por la raíz cuadrada del número de neuronas previas
        self.weights = tf.Variable(tf.random.normal(shape=[prev_num_neurons, num_neurons], stddev=tf.sqrt(2.0 / prev_num_neurons)), trainable=True)
        # Inicialización de sesgos con una media de 0.1 y una desviación estándar de 0.01
        self.biases = tf.Variable(tf.random.normal(shape=[1, num_neurons], mean=0.1, stddev=0.01), trainable=True)
        self.batch_size = batch_size
        self.prev_num_neurons = prev_num_neurons

    def update_weights_and_biases(self, new_weights, new_biases):
        """
        Método para actualizar los pesos y sesgos de la capa.

        Parámetros:
        - new_weights: Nuevos valores para los pesos.
        - new_biases: Nuevos valores para los sesgos.
        """
        self.weights.assign(new_weights)
        self.biases.assign(new_biases)

    def run(self, input):
        """
        Método que realiza la pasada hacia adelante (forward pass) en la capa.

        Parámetros:
        - input: Entrada a la capa (puede ser la salida de una capa anterior).

        Retorna:
        - output: Salida de la capa después de aplicar la función de activación.
        """
        # Calcular el output como el producto matricial de la entrada y los pesos, sumado a los sesgos
        output = tf.matmul(input, self.weights) + self.biases
        # Aplicar la función de activación si está definida
        if self.activation is not None:
            output = self.activation(output)
        return output

class FeedForwardModel:
    """
    Clase que representa un modelo feedforward de red neuronal.

    Atributos:
    - layers: Lista de capas densas (DenseLayer).
    - inputs: Entrada del modelo.
    - outputs: Salida esperada (etiquetas).
    - loss: Función de pérdida a utilizar (ej. tf.losses.MeanSquaredError).
    - learning_rate: Tasa de aprendizaje para el optimizador.
    - value_loss: Último valor de la pérdida calculada.
    - value_accuracy: Último valor de la precisión calculada.
    - r2: Último valor del coeficiente de determinación R^2.
    """

    def __init__(self, layers, inputs, outputs, loss, learning_rate):
        """
        Constructor de la clase FeedForwardModel.

        Parámetros:
        - layers: Lista de capas que componen el modelo.
        - inputs: Tensor que representa la entrada al modelo.
        - outputs: Tensor que representa la salida esperada.
        - loss: Función de pérdida a utilizar.
        - learning_rate: Tasa de aprendizaje.
        """
        self.layers = layers
        self.inputs = inputs
        self.outputs = outputs
        self.loss = loss 
        self.learning_rate = learning_rate
        self.value_loss = 0
        self.value_accuracy = 0
        self.r2 = 0
    
    def forward_pass(self, input):
        """
        Realiza la pasada hacia adelante (forward pass) a través de todas las capas del modelo.

        Parámetros:
        - input: Entrada al modelo.

        Retorna:
        - input: Salida después de pasar por todas las capas.
        """
        for layer in self.layers:
            input = layer.run(input)
        self.logits = input
        return input

    def step(self):
        """
        Realiza un paso de entrenamiento que incluye el cálculo del gradiente y la actualización de los pesos.

        Retorna:
        - loss: Valor de la pérdida calculada.
        - logits: Predicciones del modelo.
        """
        with tf.GradientTape() as tape:
            # Realizar la pasada hacia adelante y calcular la pérdida
            logits = self.forward_pass(self.inputs)
            loss = self.loss(self.outputs, logits)
        
        # Obtener los pesos y sesgos de cada capa
        weights_and_biases = [(layer.weights, layer.biases) for layer in self.layers]
        # Calcular los gradientes con respecto a la pérdida
        gradients = tape.gradient(loss, [wb for pair in weights_and_biases for wb in pair])
        
        # Actualizar los pesos y sesgos en cada capa
        for layer, grad_pair in zip(self.layers, zip(*[iter(gradients)]*2)):
            weights_grad, biases_grad = grad_pair
            new_weights = layer.weights - self.learning_rate * weights_grad
            new_biases = layer.biases - self.learning_rate * biases_grad
            layer.update_weights_and_biases(new_weights, new_biases)
        
        return loss, logits

    def train(self, epochs):
        """
        Entrena el modelo durante un número especificado de épocas.

        Parámetros:
        - epochs: Número de épocas para entrenar.
        """
        for epoch in range(epochs):
            loss, logits = self.step()
            self.value_loss = loss.numpy()
            print(f"------> Epoch {epoch+1}: Loss = {self.value_loss}")
    
    def predict(self, input):
        """
        Realiza una predicción con el modelo entrenado.

        Parámetros:
        - input: Entrada para la predicción.

        Retorna:
        - Salida del modelo después de la pasada hacia adelante.
        """
        return self.forward_pass(input)
