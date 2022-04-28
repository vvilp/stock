from utils import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import minmax_scale


# codes = ["BHP.AX", "AUDUSD%3DX", "CL%3DF"]
codes = ["BHP.AX", "AUDUSD%3DX"]
seq_array = []
for code in codes:
    prices = get_column_data_csv(f"data/{code}.csv", "Adj Close")
    # seq_array.append(get_rate_array(prices))
    seq_array.append(prices)
    # seq_array.append(minmax_scale(prices, feature_range=(0,100)))

# for i in range (5, 15 ,2):
#     avg_prices = get_pre_n_average_array(seq_array[0], i  * 2)
#     seq_array.append(avg_prices)

seq_len =  len(seq_array[0])

input_steps = 10
target_steps = 10
n_features = len(seq_array)

x, y = split_combine_seq(seq_array, seq_len, input_steps, target_steps)
last_x = get_last_n_input(seq_array, seq_len, input_steps)

print (x[-1:])
print (y[-1:])
print (last_x)

x = np.array(x)
last_x = np.array(last_x)
input = x.reshape((x.shape[0], x.shape[1], n_features))
last_x_input = last_x.reshape((last_x.shape[0], last_x.shape[1], n_features))
target = np.array(y)

print(input[-10:])
print(last_x_input)

sep_index = int(seq_len * 0.80)
# print (sep_index)
train_x, train_y = input[:sep_index], target[:sep_index]
test_x, test_y = input[sep_index:], target[sep_index:]

model = tf.keras.Sequential()
model.add(layers.GRU(n_features * 5, activation='relu', input_shape=(input_steps, n_features)))
model.add(layers.Dense(target_steps * 2 , activation='relu'))
model.add(layers.Dense(target_steps))
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001), loss=tf.keras.losses.Huber())
h = model.fit(train_x,train_y, epochs=200)

p_train_y = model.predict(train_x)
p_test_y = model.predict(test_x)
p_last = model.predict(last_x_input)

mse = (np.square(test_y - p_test_y)).mean(axis=1)
# print (mse)
# print (np.average(mse))

label_array = ['origin', 'predict']
plot_x_array = [range(0,len(train_y[:,0])), range(0,len(p_train_y[:,0])) ]
plot_y_array = [train_y[:,0], p_train_y[:,0]]
plot(plot_x_array, plot_y_array, label_array, "plot/train_res.png")

plot_x_array = [range(0,len(test_y[:,0])), range(0,len(p_test_y[:,0])) ]
plot_y_array = [test_y[:,0], p_test_y[:,0]]
plot(plot_x_array, plot_y_array, label_array, "plot/test_res.png")

plot_x_array = [range(0,len(test_y[-1])), range(0,len(p_test_y[-1])) ]
plot_y_array = [test_y[-1], p_test_y[-1]]
plot(plot_x_array, plot_y_array, label_array, "plot/predict_last_n_days.png")

plot_x_array = range(0,len(p_last[0]))
plot_y_array = p_last[0]
plot_simple(plot_x_array, plot_y_array, "plot/predict_next_n_days.png")

plot_x_arrays = []
plot_y_arrays = []
for i in range (8):
    index = -i * target_steps-1
    plot_x_arrays.append([range(0,len(test_y[index])), range(0,len(p_test_y[index])) ])
    plot_y_arrays.append([test_y[index], p_test_y[index]])
plot_sub(plot_x_arrays, plot_y_arrays, label_array, "plot/predict_last_n_days_multiple.png")

loss = h.history['loss']
plot_x_array = [range(0,len(loss)) ]
plot_y_array = [loss]
plot(plot_x_array, plot_y_array, ["loss"], "plot/loss.png")

