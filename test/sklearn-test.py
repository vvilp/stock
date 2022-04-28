from utils import * 
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale

seq_array = get_column_data_csv("data/SUN.AX.csv", "Adj Close")
seq_array_scaled = minmax_scale(seq_array, feature_range=(0,1))


label_array = ['origin']
plot_x_array = [range(0,len(seq_array_scaled)) ]
plot_y_array = [seq_array_scaled]
plot(plot_x_array, plot_y_array, label_array, "plot/sklean-scale.png")