import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import csv
import matplotlib.pyplot as plt

# test = np.array([[0.7227243],
#                  [0.72455484],
#                  [0.72762626],
#                  [0.7297867],
#                  [0.73342997],
#                  [0.7369942],
#                  [0.7404718],
#                  [0.7421414],
#                  [0.7457617],
#                  [0.7465288]])

# print (test)
# test = test.ravel()
# print (test)
def splitSequence(seq, n_steps):
    X = []
    y = []
    for i in range(len(seq)):
        lastIndex = i + n_steps
        if lastIndex > len(seq) - 1:
            break
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        X.append(seq_X)
        y.append(seq_y)
        pass
    X = np.array(X)
    y = np.array(y)
    return X,y 
    
price_close = []
with open('AUDUSD.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Adj Close'] != 'null':
            price_close.append(float(row['Adj Close']))

n_steps = 5
n_features = 1
X, y = splitSequence(price_close, n_steps);

print (X[:10])
print (y[:10])
input = X.reshape((X.shape[0], X.shape[1], n_features))


model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(5))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())
model.fit(input, y, epochs=50)

predict_y = model.predict(input, verbose=1)

print (predict_y[0:10])

fig = plt.figure()
ax = fig.add_subplot(111)
x = range(0,len(y))
ax.plot(x, y, linewidth=0.5)

y1 = predict_y.ravel()
x1 = range(0,len(y1))
ax.plot(x1, y1, linewidth=0.5)
plt.savefig('res.pdf')