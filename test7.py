from utils import * 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import minmax_scale


codes = ['ALL.AX' ,'GMG.AX' , 'ANZ.AX', 'BHP.AX', 'SUN.AX' ,  'CSL.AX', 'FMG.AX']
seq_array = []
# seq_array.append(get_rate_array(get_column_data_csv("data/SUN.AX.csv", "Adj Close")))
for code in codes:
    prices = get_column_data_csv(f"data/{code}.csv", "Adj Close")
    # seq_array.append(get_rate_array(prices))
    # seq_array.append(prices)
    seq_array.append(minmax_scale(prices, feature_range=(0,100)))

seq_len =  len(seq_array[0])


input_steps = 15
target_steps = 10
n_features = len(seq_array)

x, y = split_combine_seq(seq_array, seq_len, input_steps, target_steps)

x = np.array(x)
input = x.reshape((x.shape[0], x.shape[1], n_features))
print(input[:10])
target = np.array(y)
print(target[:10])
print(target[:10,0])

sep_index = int(seq_len * 0.80)
print (sep_index)
train_x, train_y = input[:sep_index], target[:sep_index]
test_x, test_y = input[sep_index:], target[sep_index:]



model = tf.keras.Sequential()
model.add(layers.GRU(n_features * 10, activation='relu', input_shape=(input_steps, n_features)))
model.add(layers.Dense(n_features * 2 , activation='relu'))
model.add(layers.Dense(target_steps))
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.Huber())
h = model.fit(train_x,train_y, epochs=50)

p_train_y = model.predict(train_x)

p_test_y = model.predict(test_x)

print (test_y[0])
print (p_test_y[0])

mse = (np.square(test_y - p_test_y)).mean(axis=1)
print (mse)
print (np.average(mse))

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

plot_x_arrays = []
plot_y_arrays = []
for i in range (8):
    index = -i * target_steps-1
    plot_x_arrays.append([range(0,len(test_y[index])), range(0,len(p_test_y[index])) ])
    plot_y_arrays.append([test_y[index], p_test_y[index]])
plot_sub(plot_x_arrays, plot_y_arrays, label_array, "plot/res_p_last_n.png")

loss = h.history['loss']
loss = loss[50:]
plot_x_array = [range(0,len(loss)) ]
plot_y_array = [loss]
plot(plot_x_array, plot_y_array, ["loss"], "plot/loss.png")

