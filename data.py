import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class CarFollowingData(
    Dataset):  # raw_data.size = (num_event, dim_state, num_time_step)  states:(spacing, self_speed, relative_speed, lead_speed)
    def __init__(self, data_path, historical_len, predict_len, max_len, Ts, rolling_window):
        self.files = os.listdir(data_path)
        self.max_len = max_len
        self.historical_len = historical_len
        self.predict_len = predict_len
        self.Ts = Ts
        self.rolling_window = rolling_window
        self.raw_data_list = []
        self.data_list = []
        self.acc_metric_list = []
        for f in self.files:
            if f.endswith(".npy"):
                self.raw_data_list.append(np.load(os.path.join(data_path, f)))
        self.raw_data = np.concatenate(self.raw_data_list, axis=0)
        if self.raw_data.shape[2] > max_len:
            self.raw_data = self.raw_data[:, :, :max_len]
        assert historical_len + predict_len <= self.raw_data.shape[2]
        assert historical_len >= predict_len
        for i in range(self.raw_data.shape[0]):
            for j in range(self.raw_data.shape[2] - (historical_len + predict_len) + 1):
                self.data_list.append(self.raw_data[i, :, j:(j + historical_len + predict_len)])
        self.data = np.array(self.data_list)
        for k in range(self.data.shape[0]):
            for n in range(historical_len + predict_len - self.rolling_window + 1):
                segment_speed = self.data[k, 1, n:n + self.rolling_window]
                segment_acc = np.abs(np.diff(segment_speed) / self.Ts)
                acc_mean = np.mean(segment_acc)
                acc_std = np.std(segment_acc)
                acc_coef = acc_std / acc_mean
                self.acc_metric_list.append(acc_coef)
        acc_metric = np.array(self.acc_metric_list)
        self.acc_metric_max = np.quantile(acc_metric, 0.85)
        print(self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def calculate_acc_metric(self, speed_trajectory):
        traject_len = speed_trajectory.shape[0]
        assert traject_len >= self.rolling_window
        acc_metric_list = []
        for i in range(traject_len - self.rolling_window + 1):
            segment_speed = speed_trajectory[i:i + self.rolling_window]
            segment_acc = np.abs(np.diff(segment_speed) / self.Ts)
            mean_segment = np.mean(segment_acc)
            std_segment = np.std(segment_acc)
            coef_segment = std_segment / mean_segment
            acc_metric_list.append(coef_segment)
        acc_metric = np.mean(acc_metric_list)
        normalized_acc_metric = acc_metric / self.acc_metric_max
        return normalized_acc_metric

    def __getitem__(self, idx: int):
        event = self.data[idx]  # a car-following event
        spacing = event[0, :]  # spacing between self vehicle (sv) and leading vehicle (lv)
        self_v = event[1, :]  # speed of sv
        delta_v = event[2, :]  # relative speed between sv and lv
        lead_v = event[3, :]  # speed of lv
        historical_sv = self_v[:self.historical_len]
        future_sv = self_v[self.historical_len:]
        historical_dv = delta_v[:self.historical_len]
        historical_ds = spacing[:self.historical_len]
        future_lv = lead_v[self.historical_len:]
        assert len(historical_sv) == self.historical_len
        assert len(historical_dv) == self.historical_len
        assert len(historical_ds) == self.historical_len
        assert len(future_lv) == self.predict_len
        assert len(future_sv) == self.predict_len
        historical_input = np.transpose(np.array([historical_ds, historical_sv, historical_dv]))
        assert historical_input.shape[0] == self.historical_len
        assert historical_input.shape[1] == 3
        acc_metric = self.calculate_acc_metric(future_sv)
        return {'historical_input': historical_input, 'future_lv': future_lv, 'future_sv': future_sv,
                'style_metric': acc_metric}

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        historical_input_list = []
        future_lv_list = []
        future_sv_list = []
        style_metric_list = []
        for item in batch:
            historical_input_list.append(item['historical_input'])
            future_lv_list.append(item['future_lv'])
            future_sv_list.append(item['future_sv'])
            style_metric_list.append(item['style_metric'])
        historical_input_batch = torch.Tensor(np.array(historical_input_list)).type(torch.float)
        future_lv_batch = torch.Tensor(np.array(future_lv_list)).type(torch.float)
        future_sv_batch = torch.Tensor(np.array(future_sv_list)).type(torch.float)
        style_metric_batch = torch.Tensor(np.array(style_metric_list)).type(torch.float)

        future_lv_batch = torch.unsqueeze(future_lv_batch, -1)
        future_sv_batch = torch.unsqueeze(future_sv_batch, -1)
        style_metric_batch = torch.unsqueeze(style_metric_batch, -1)

        assert historical_input_batch.shape[0] == batch_size
        assert future_lv_batch.shape[0] == batch_size
        assert future_sv_batch.shape[0] == batch_size
        assert style_metric_batch.shape[0] == batch_size
        batch_output = {'batch_size': batch_size, 'historical_input': historical_input_batch,
                        'future_lv': future_lv_batch, 'future_sv': future_sv_batch, 'style_metric': style_metric_batch}
        return batch_output


def getDataLoader(data_path, batch_size, historical_len, predict_len, max_len, Ts, rolling_window):
    dataset = CarFollowingData(data_path, historical_len, predict_len, max_len, Ts, rolling_window)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=CarFollowingData.collate_fn, drop_last=True)
    return dataloader
