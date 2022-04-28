import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import csv
import matplotlib.pyplot as plt

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
price_pre = 0
price_change = 0
with open('AUDUSD.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Adj Close'] != 'null':
            if price_pre == 0:
                price_change = 0.0
                price_pre = float(row['Adj Close'])
            else:
                price_change = ((float(row['Adj Close']) - price_pre) / price_pre)
                price_change = price_change * 100.0
                price_pre = float(row['Adj Close'])
            price_close.append(price_change)
        else:
            price_close.append(0)

print (price_close[:10])

n_steps = 5
n_features = 1
X, y = splitSequence(price_close, n_steps);

print (X[:10])
print (y[:10])
input = X.reshape((X.shape[0], X.shape[1], n_features))


model = tf.keras.Sequential()
model.add(layers.LSTM(25,activation='elu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(10, activation='elu'))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())
model.fit(input, y, epochs=500, batch_size = 20)

predict_y = model.predict(input, verbose=1)

print (predict_y[0:10])

fig = plt.figure()
ax = fig.add_subplot(111)
x = range(0,len(y[:500]))
ax.bar(x, y[:500])

y1 = predict_y.ravel()
x1 = range(0,len(y1[:500]))
ax.plot(x1, y1[:500])
plt.savefig('res.pdf', dpi=300)