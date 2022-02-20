import csv
import enum
import matplotlib.pyplot as plt

def get_column_data_csv(file_name, col_name):
    res = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            o = row[col_name]
             # if obj is null, use pre obj
            i = float(o) if o!= 'null' and o!='0' and o!= 0 else res[-1]
            res.append(i)
    return res

def get_rate_array(origin_array):
    res = []
    for i, o in enumerate(origin_array):
        c_rate = 0.0 if i == 0 else (o - origin_array[i-1]) / origin_array[i-1]
        res.append(c_rate)
    return res

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
    return X,y 

def split_combine_seq(seq_array, seq_len, n_steps, target_steps):
    # input: seq_array > [seq1, seq2, ..., ] -> [ [....], [....] ]
    # output: 
    # x
    # [ 
    #   [[seq1_x1, seq2_x1], [....]]
    #   [[seq1_x2, seq2_x2], [....]]
    # ]
    # y
    # [
    #    [y,y]
    #    [y,y]
    # ]
    #
    #
    x_array = []
    y_array = []
    for i in range(0, seq_len):
        x = []
        for step in range(0, n_steps):
            feature_array = []
            for iii, seq in enumerate(seq_array):
                feature_array.append(seq[i + step])
            x.append(feature_array)
        x_array.append(x)

        target_array = []
        for step in range(0, target_steps):
            target_array.append(seq_array[0][i + n_steps + step])
        y_array.append(target_array)

        if i + n_steps + target_steps == seq_len:
            break
    return x_array , y_array
        


def plot(x_array, y_array, label_array, file_name):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for i, o in enumerate(x_array):
        x = x_array[i]
        y = y_array[i]
        ax.plot(x, y, label = label_array[i])
    ax.legend(loc='best')
    plt.savefig(file_name, dpi=300)