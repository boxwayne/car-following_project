import modules
import numpy as np
import torch
from torch import nn
import model
import data

#train_data = np.random.randn(50, 4, 100)
#np.save("../car-following project/train_data.npy", train_data)
dataloader = data.getDataLoader("/home/bld/car-following_zijun/car-following_project", batch_size=8, historical_len=10, predict_len=5, max_len=100, Ts=0.1, rolling_window=5)

models = model.Model(3, 64, 15)
for item in dataloader:
    a = item['historical_input']
    b = item['future_lv']
    c = item['style_metric']
    d = models(a, c, b)
    print(d.shape)


