from utils import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n_steps = 5
n_features = 1
value_array = get_column_data_csv("SUN.AX.csv", "Adj Close")
print (value_array[:10])
rate_array = get_rate_array(value_array)
rate_array = [ each * 100 for each in rate_array ]
print (rate_array[:10])
x , y = splitSequence(value_array, n_steps)
x , y = np.array(x) , np.array(y)
print (x[:10])
print (y[:10])

input = x.reshape((x.shape[0], x.shape[1], n_features))
model = tf.keras.Sequential()
model.add(layers.LSTM(20, activation='elu', input_shape=(n_steps, n_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())
h = model.fit(input, y, epochs=500, batch_size = 20)

predict_y = model.predict(input, verbose=1)

print (predict_y[:10])

plot_x_array = [range(0,len(y)) , range(0,len(predict_y.ravel())) ]
plot_y_array = [y, predict_y.ravel()]
plot(plot_x_array, plot_y_array, "res.png")

loss = h.history['loss']
plot_x_array = [range(0,len(loss)) ]
plot_y_array = [loss]
plot(plot_x_array, plot_y_array, "loss.png")
