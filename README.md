
# TensorFlowFromScratch-NNBuilder

A simple implementation of a neural network built from scratch using TensorFlow, without pre-built APIs. This repository covers the basic concepts of neural networks, including dense layers, forward propagation, and training loops.

## Features

- Custom dense layers with activation functions.
- Manual implementation of forward propagation.
- Training loop with gradient descent.
- Easy to extend and modify for learning and experimentation.

## Getting Started

### Prerequisites

Make sure you have Python installed, along with TensorFlow and NumPy:

```bash
pip install tensorflow numpy
```

### Running the Code

1. **Initialize a Dense Layer:**

   ```python
   from model import DenseLayer
   
   # Create a dense layer with 3 neurons and ReLU activation
   layer = DenseLayer(num_neurons=3, activation=tf.nn.relu, prev_num_neurons=2)
   ```

2. **Create a FeedForward Model:**

   ```python
   from model import FeedForwardModel

   # Example input and output data (for simplicity, using random data here)
   inputs = tf.random.normal([5, 2])
   outputs = tf.random.normal([5, 3])

   # Define the layers
   layers = [
       DenseLayer(num_neurons=3, activation=tf.nn.relu, prev_num_neurons=2),
       DenseLayer(num_neurons=2, activation=tf.nn.softmax, prev_num_neurons=3)
   ]

   # Define the model with Mean Squared Error loss
   model = FeedForwardModel(layers=layers, inputs=inputs, outputs=outputs, loss=tf.losses.MeanSquaredError(), learning_rate=0.01)
   ```

3. **Train the Model:**

   ```python
   # Train the model for 10 epochs
   model.train(epochs=10)
   ```

4. **Make Predictions:**

   ```python
   # Make a prediction with new input data
   new_input = tf.random.normal([1, 2])
   prediction = model.predict(new_input)
   print(f"Prediction: {prediction.numpy()}")
   ```

## Example Output

After running the training loop, you should see output similar to:

```
------> Epoch 1: Loss = 0.4321
------> Epoch 2: Loss = 0.4105
...
------> Epoch 10: Loss = 0.2547
```

## Customization

You can easily customize the model by:

- Changing the number of neurons in each layer.
- Using different activation functions.
- Modifying the loss function or learning rate.

Experiment with different configurations to see how they affect the model's performance!

## License

This project is open-source and available under the MIT License.
