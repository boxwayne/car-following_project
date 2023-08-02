import modules
import numpy as np
import torch
from torch import nn
import model
import data

#train_data = np.random.randn(50, 4, 100)
#np.save("../car-following project/train_data.npy", train_data)
dataloader = data.getDataLoader("Users/hezijun/Desktop/car-following project/train_data.npy", batch_size=5, historical_len=10, predict_len=5, max_len=100, Ts=0.1, rolling_window=5)

for item in dataloader:
    print(item.size())


