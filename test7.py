from utils import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n_steps = 15
target_steps = 10
n_features = 2

a1 = get_column_data_csv("data/SUN.AX.csv", "Adj Close")
print(a1[:10])

a2 = get_column_data_csv("data/AXJO.csv", "Adj Close")
print(a2[:10])

seq_array = [a1, a2]
seq_len =  len(a1)

x, y = split_combine_seq(seq_array, seq_len, n_steps, target_steps)

x = np.array(x)
input = x.reshape((x.shape[0], x.shape[1], n_features))
print(input[:10])
target = np.array(y)
print(target[:10])
print(target[:10,0])

sep_index = int(seq_len * 0.9)
print (sep_index)
train_x, train_y = input[:sep_index], target[:sep_index]
test_x, test_y = input[sep_index:], target[sep_index:]

print (test_y[-1])

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(10))
model.add(layers.Dense(target_steps))
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.MeanSquaredError())
h = model.fit(train_x,train_y, epochs=500)

p_train_y = model.predict(train_x)

p_test_y = model.predict(test_x)

label_array = ['origin', 'predict']
plot_x_array = [range(0,len(train_y[:,0])), range(0,len(p_train_y[:,0])) ]
plot_y_array = [train_y[:,0], p_train_y[:,0]]
plot(plot_x_array, plot_y_array, label_array, "plot/res.png")

plot_x_array = [range(0,len(test_y[:,0])), range(0,len(p_test_y[:,0])) ]
plot_y_array = [test_y[:,0], p_test_y[:,0]]
plot(plot_x_array, plot_y_array, label_array, "plot/res_p.png")

plot_x_array = [range(0,len(test_y[-1])), range(0,len(p_test_y[-1])) ]
plot_y_array = [test_y[-1], p_test_y[-1]]
plot(plot_x_array, plot_y_array, label_array, "plot/res_p_last.png")

loss = h.history['loss']
plot_x_array = [range(0,len(loss)) ]
plot_y_array = [loss]
plot(plot_x_array, plot_y_array, ["loss"], "plot/loss.png")