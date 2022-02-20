import matplotlib.pyplot as plt
import csv


price_close = []
with open('data/AUDUSD.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Adj Close'] != 'null':
            price_close.append(float(row['Adj Close']))

labels = ["origin" , "predict"]
fig = plt.figure()
ax = fig.add_subplot(111)
x = range(0,len(price_close))
ax.plot(x, price_close, label=labels[0])

price_close1 = [x + 0.1 for x in price_close]
ax.plot(x, price_close1 , label=labels[1])
ax.legend(loc='best')
plt.savefig("test.png", dpi=300)