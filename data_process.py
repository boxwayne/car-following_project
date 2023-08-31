import data
import numpy as np
import matplotlib.pyplot as plt

train_dataset_path = "/home/bld/Download/HighD/train_data/HighD_splited_train_data.npy"
validation_dataset_path = "/home/bld/Download/HighD/val_data/HighD_splited_val_data.npy"
test_dataset_path = "/home/bld/Download/HighD/test_data/HighD_splited_test_data.npy"

historical_len = 190  # number of time step in historical car-following states
predict_len = 160  # number of time step in predicted future speed profile of self vehicle
max_len = 375
rolling_window = 160  # number of time step of a given speed trajectory to evaluate the style metric
Ts = 0.04  # sampling time step of data

dataset = data.CarFollowingData(train_dataset_path, historical_len, predict_len, max_len, Ts, rolling_window)
style_metric_list = dataset.acc_metric_list
plt.hist(style_metric_list, bins=60, color='skyblue')
plt.title('style metric value histogram')
plt.xlabel('style value')
plt.ylabel('frequency')
plt.show()