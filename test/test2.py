import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

n_steps = 3
n_features = 2
X = np.array([  [[1, 1], [2, 2], [3, 3]], 
                [[2, 2], [3, 3], [4, 4]], 
                [[3, 3], [4, 4], [5, 5]], 
                [[9, 9], [10, 10], [11, 11]], 
                [[50, 50], [51, 51], [52, 52]]])
# y = np.array([4, 5, 6, 12, 53])
y = np.array([[4, 6], [5, 7], [6, 8], [12, 14], [53, 55]])
print(X)
print(y)

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu',
          input_shape=(n_steps, n_features)))
# model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=(n_steps, n_features)))
# model.add(layers.Bidirectional(layers.LSTM(50)))
model.add(layers.Dense(10))
model.add(layers.Dense(2))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

model.fit(X, y, epochs=200)

predictNextNumber = model.predict(np.array([[[6, 6], [7, 7], [8, 8]], [[4, 4], [5, 5], [6, 6]], [[20, 20], [21, 21], [22, 22]], [[100, 100], [101, 101], [102, 102]]]), verbose=1)
print(predictNextNumber)
