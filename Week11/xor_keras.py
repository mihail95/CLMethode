"""
Implementation des XOR-Beispiels mittels Keras
"""

#https://keras.io/getting_started/
#If you install TensorFlow, critically, you should reinstall Keras 3 afterwards.
# This is a temporary step while TensorFlow is pinned to Keras 2, and will no longer be necessary after TensorFlow 2.16.
# The cause is that tensorflow==2.15 will overwrite your Keras installation with keras==2.15.
# pip install tensorflow
# pip install --upgrade keras 

# Anaconda: conda install -c anaconda keras

# Quellen (auch für Kommentare):
#https://blog.thoughtram.io/machine-learning/2016/11/02/understanding-XOR-with-keras-and-tensorlow.html
#https://keras.io/guides
#https://keras.io/api

import numpy as np
import keras
from keras.layers import Dense

# Reproducibility: https://keras.io/examples/keras_recipes/reproducibility_recipes/
#keras.utils.set_random_seed(812)


# the four different states of the XOR gate
# (= a matrix with four rows and two columns)
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
# (= a matrix with four rows and one column)
target_data = np.array([[0],[1],[1],[0]], "float32")

#A Sequential model is appropriate for a plain stack of layers
# where each layer has exactly one input tensor and one output tensor.
# Alternative: The Keras functional API is a way to create models that are more flexible
# than the keras.Sequential API. The functional API can handle models
# with non-linear topology, shared layers, and even multiple inputs or outputs.
# Tensors are multi-dimensional arrays with a uniform type. Tensors are (kind of) like np.arrays.
# All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.
# (https://www.tensorflow.org/guide/tensor)
model = keras.Sequential()

# In general, it's a recommended best practice to always specify the input shape of a Sequential model in advance if you know what it is.
# A shape tuple (tuple of integers or None objects), not including the batch size. For instance, shape=(32,) indicates
# that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None;
# None elements represent dimensions where the shape is not known and may vary (e.g. sequence length).
model.add(keras.Input(shape=(2,)))


#Just your regular densely-connected NN layer.
# Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function
# passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer
# (only applicable if use_bias is True (default)).
# units: Positive integer, dimensionality of the output space.
model.add(Dense(units = 2, activation='relu'))
model.add(Dense(units = 1, activation='sigmoid'))

# Frage: Warum 9 Parameter?
#print(model.summary())


# see https://keras.io/api/layers/core_layers/dense/ about initial weights
#print()
#print("Initial weights (random):")
#print(model.get_weights())

# The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
# optimizer SGD = stochastic gradient descent method
# Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model.
# Note that you may use any loss function as a metric.
model.compile(loss='binary_crossentropy',
              optimizer= keras.optimizers.SGD(learning_rate=0.01),
              metrics=['binary_accuracy'])

#verbose: "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
model.fit(training_data, target_data, epochs=1000, verbose=0)

#print()
#print("Predictions:")
#print(model.predict(training_data, verbose=0).round(2))

#print("Evaluation:")
#print(model.evaluate(training_data, target_data, verbose=1))

# Mal sind die Predictions richtig und mal falsch (zumindest wenn die Anzahl der Epochen groß genug ist)
# -> Zufallsinitialisierung der Gewichte!
# Das Gradientenverfahren kann in einem lokalen Minimum feststecken