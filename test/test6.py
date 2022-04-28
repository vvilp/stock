from utils import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n_steps = 3
target_steps = 3
n_features = 2

a1 = [1,2,3,4,5,6,7,8,9]
a2 = [11,12,13,14,15,16,17,18,19]
seq_array = [a1, a2]

x, y = split_combine_seq(seq_array, len(a1), n_steps,target_steps )

print (a1)
print (a2)
print (x)
print (y)
x = np.array(x)
input = x.reshape((x.shape[0], x.shape[1], n_features))
print (input)
target = np.array(y)
print (target)

print (input[:2])
print (target[:2])

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())
model.fit(input,target, epochs=200)

p_y = model.predict(input)
print (p_y)