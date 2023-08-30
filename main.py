import modules
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import model
import data

train_dataset_path = "/home/bld/Desktop/HighD/train_data"
validation_dataset_path = "/home/bld/Desktop/HighD/val_data"
test_dataset_path = "/home/bld/Desktop/HighD/test_data"
batch_size = 16
num_epochs = 50
historical_len = 150  # number of time step in historical car-following states
predict_len = 150  # number of time step in predicted future speed profile of self vehicle
max_len = 375
rolling_window = 125  # number of time step of a given speed trajectory to evaluate the style metric
Ts = 0.04  # sampling time step of data
train_dataloader, train_acc_metric_max, train_acc_metric_min = data.getDataLoader(train_dataset_path, batch_size,
                                                                                  historical_len, predict_len, max_len,
                                                                                  Ts, rolling_window)
validation_dataloader, validation_acc_metric_max, validation_acc_metric_min = data.getDataLoader(
    validation_dataset_path, batch_size, historical_len, predict_len, max_len, Ts, rolling_window)
test_dataloader, test_acc_metric_max, test_acc_metric_min = data.getDataLoader(test_dataset_path, batch_size,
                                                                               historical_len, predict_len, max_len, Ts,
                                                                               rolling_window)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 1e-3
saved_model_path = "/home/bld/car-following_zijun/saved_model/best_model.pt"
dim_CF_state = 3  # spacing, self_v, rel_v
dim_hidden_state = 64
num_frequency_encoding = 15
CF_model = model.Model(dim_CF_state, dim_hidden_state, num_frequency_encoding).to(device)

model_optimizer = optim.Adam(CF_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
style_metric_loss_weight = 10.0

train_loss_his = []
validation_error_his = []
best_validation_error = None
style_metric_range_eval = np.linspace(0.0, 1.0, num=11)

print("---")
for epoch in tqdm(range(num_epochs)):
    train_losses = []
    CF_model.train()
    print('training...')
    for i, item in enumerate(train_dataloader):
        if i % 10000 == 0:
            print('epoch: {0}, batch: {1}'.format(epoch, i))
        historical_CF_state = item['historical_input'].to(device)
        driving_style_metric = item['style_metric'].to(device)
        future_lead_speed = item['future_lv'].to(device)
        future_self_speed = item['future_sv'].to(device)
        try:
            predict_self_speed = CF_model(historical_CF_state, driving_style_metric, future_lead_speed)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
        predict_style_metric = modules.styleMetricEvaluation(predict_self_speed, rolling_window, Ts,
                                                             train_acc_metric_max, train_acc_metric_min)
        loss = criterion(future_self_speed, predict_self_speed) + style_metric_loss_weight * criterion(
            driving_style_metric, predict_style_metric)
        model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(CF_model.parameters(), 0.25)
        model_optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    train_loss_his.append(train_loss)
    print("Epoch: {0}| Train Loss: {1:.7f}".format(epoch + 1, train_loss))

    CF_model.eval()
    validation_errors = []

    print('evaluating...')
    for i, item in enumerate(validation_dataloader):
        if i % 8000 == 0:
            print('epoch: {0}, batch: {1}'.format(epoch, i))
        historical_CF_state = item['historical_input'].to(device)
        driving_style_metric = item['style_metric'].to(device)
        future_lead_speed = item['future_lv'].to(device)
        future_self_speed = item['future_sv'].to(device)
        try:
            predict_self_speed = CF_model(historical_CF_state, driving_style_metric, future_lead_speed)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
        reproduced_trajectory_error = criterion(future_self_speed, predict_self_speed)
        predict_style_metric = modules.styleMetricEvaluation(predict_self_speed, rolling_window, Ts,
                                                             validation_acc_metric_max, validation_acc_metric_min)
        reproduced_style_metric_error = style_metric_loss_weight * criterion(driving_style_metric, predict_style_metric)
        reproduced_output_error = reproduced_trajectory_error + reproduced_style_metric_error
        generative_style_metric_errors = []
        for value in style_metric_range_eval:
            designated_style_metric = value * torch.ones(driving_style_metric.shape, requires_grad=True).cuda()
            generative_self_speed = CF_model(historical_CF_state, designated_style_metric, future_lead_speed)
            generative_style_metric = modules.styleMetricEvaluation(generative_self_speed, rolling_window, Ts,
                                                                    validation_acc_metric_max,
                                                                    validation_acc_metric_min)
            generative_style_metric_error = style_metric_loss_weight * criterion(designated_style_metric,
                                                                                 generative_style_metric)
            generative_style_metric_errors.append(generative_style_metric_error.item())
        generative_style_metric_averaged_error = np.mean(generative_style_metric_errors)
        total_error = reproduced_output_error.item() + generative_style_metric_averaged_error
        validation_errors.append(total_error)
    validation_error = np.mean(validation_errors)
    if best_validation_error is None or best_validation_error > validation_error:
        best_validation_error = validation_error
        torch.save(CF_model, saved_model_path)

    validation_error_his.append(validation_error)
    print("Epoch: {0}| Validation error: {1:.7f}".format(epoch + 1, validation_error))


test_model = torch.load(saved_model_path).to(device)
test_model.eval()
test_errors = []

print('testing...')
for i, item in enumerate(test_dataloader):
    if i % 8000 == 0:
        print('batch: {0}'.format(i))
    historical_CF_state = item['historical_input'].to(device)
    driving_style_metric = item['style_metric'].to(device)
    future_lead_speed = item['future_lv'].to(device)
    future_self_speed = item['future_sv'].to(device)
    try:
        predict_self_speed = test_model(historical_CF_state, driving_style_metric, future_lead_speed)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            else:
                raise exception
    reproduced_trajectory_error = criterion(future_self_speed, predict_self_speed)
    predict_style_metric = modules.styleMetricEvaluation(predict_self_speed, rolling_window, Ts, test_acc_metric_max,
                                                         test_acc_metric_min)
    reproduced_style_metric_error = style_metric_loss_weight * criterion(driving_style_metric, predict_style_metric)
    reproduced_output_error = reproduced_trajectory_error + reproduced_style_metric_error
    generative_style_metric_errors = []
    for value in style_metric_range_eval:
        designated_style_metric = value * torch.ones(driving_style_metric.shape, requires_grad=True).cuda()
        generative_self_speed = test_model(historical_CF_state, designated_style_metric, future_lead_speed)
        generative_style_metric = modules.styleMetricEvaluation(generative_self_speed, rolling_window, Ts,
                                                                test_acc_metric_max, test_acc_metric_min)
        generative_style_metric_error = style_metric_loss_weight * criterion(designated_style_metric,
                                                                             generative_style_metric)
        generative_style_metric_errors.append(generative_style_metric_error.item())
    generative_style_metric_averaged_error = np.mean(generative_style_metric_errors)
    total_error = reproduced_output_error.item() + generative_style_metric_averaged_error
    test_errors.append(total_error)
test_error = np.mean(test_errors)
print('test_error: {0:.7f}'.format(test_error))
