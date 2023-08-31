import modules
import numpy as np
import torch
import model
import data
import matplotlib.pyplot as plt

model_path = "/home/bld/桌面/car_following_zijun/saved_model/best_model.pth"
test_dataset_path = "/home/bld/Download/HighD/test_data/HighD_splited_test_data.npy"
batch_size = 1
historical_len = 190  # number of time step in historical car-following states
predict_len = 160  # number of time step in predicted future speed profile of self vehicle
max_len = 375
rolling_window = 160  # number of time step of a given speed trajectory to evaluate the style metric
Ts = 0.04  # sampling time step of data

test_dataloader, test_acc_metric_max, test_acc_metric_min = data.getDataLoader(test_dataset_path, batch_size,
                                                                               historical_len, predict_len, max_len, Ts,
                                                                               rolling_window)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model = torch.load(model_path).to(device)
test_model.eval()

total_count = 0
crash_count = 0
style_metric_range_eval = np.array([0.0, 0.5, 1.0])
plot_list = []
for i, item in enumerate(test_dataloader):
    if i % 8000 == 0:
        print('batch: {0}'.format(i))
    historical_CF_state = item['historical_input'].to(device)
    driving_style_metric = item['style_metric'].to(device)
    future_lead_speed = item['future_lv'].to(device)
    future_self_speed = item['future_sv'].to(device)

    is_crashed = False
    future_states_list = []
    metrics_list = []
    for value in style_metric_range_eval:
        designated_style_metric = value * torch.ones(driving_style_metric.shape, requires_grad=True).cuda()
        try:
            predict_self_speed = test_model(historical_CF_state, designated_style_metric, future_lead_speed)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
        predicted_style_metric = modules.styleMetricEvaluation(predict_self_speed, rolling_window, Ts,
                                                                test_acc_metric_max, test_acc_metric_min)
        metrics_list.append(torch.squeeze(predicted_style_metric).item())
        current_states = torch.squeeze(historical_CF_state[:, -1, :]).clone().detach().cpu().numpy()     # (initial spacing, initial self speed, initial ref speed)
        future_dv = torch.squeeze(future_lead_speed - predict_self_speed).clone().detach().cpu().numpy()
        future_states = []
        future_states.append(current_states)
        for i in range(len(future_dv)):
            ds = (future_dv[i] + future_states[i][2]) / 2.0 * Ts + future_states[i][0]
            sv = predict_self_speed[:, i, :].item()
            dv = future_dv[i]
            future_states.append([ds, sv, dv])
        future_states = np.array(future_states)
        future_states_list.append(future_states)
        future_ds = future_states[:, 0]
        if min(future_ds) < 0.1:
            is_crashed = True

    metric_error = np.mean(np.abs(np.array(metrics_list) - style_metric_range_eval))
    metrics_list = np.array(metrics_list)
    future_states_list = np.array(future_states_list)
    future_lv = torch.squeeze(future_lead_speed).clone().detach().cpu().numpy()
    plot_example = {'metric_error': metric_error, 'metrics_list': metrics_list, 'future_states_list': future_states_list, 'future_lv': future_lv}

    if is_crashed == True:
        crash_count += 1
        print('crashed event detected: {0}'.format(crash_count))
    else:
        if len(plot_list) < 3:
            plot_list.append(plot_example)
        elif plot_list[2]['metric_error'] > plot_example['metric_error']:
            plot_list[2] = plot_example
        sorted(plot_list, key=lambda x: x['metric_error'])
    total_count += 1
print('Test finished. Total number of events: {0}. Number of crashed events: {1}. Plot three examples with lowest error'.format(total_count, crash_count))

for i in range(3):
    example = plot_list[i]
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(example['future_lv'], label='lv speed')
    plt.plot(example['future_states_list'][0, :, 1], label='sv speed')
    plt.title('designated metric: {0:.7f}, metric error: {1:.7f}'.format(example['metrics_list'][0], example['metric_error']))
    plt.legend()
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')
    plt.subplot(2, 3, 4)
    plt.plot(example['future_states_list'][0, :, 0], label='spacing')
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')

    plt.subplot(2, 3, 2)
    plt.plot(example['future_lv'], label='lv speed')
    plt.plot(example['future_states_list'][1, :, 1], label='sv speed')
    plt.title('designated metric: {0:.7f}, metric error: {1:.7f}'.format(example['metrics_list'][1], example['metric_error']))
    plt.legend()
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')
    plt.subplot(2, 3, 5)
    plt.plot(example['future_states_list'][1, :, 0], label='spacing')
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')

    plt.subplot(2, 3, 3)
    plt.plot(example['future_lv'], label='lv speed')
    plt.plot(example['future_states_list'][2, :, 1], label='sv speed')
    plt.title('designated metric: {0:.7f}, metric error: {1:.7f}'.format(example['metrics_list'][2], example['metric_error']))
    plt.legend()
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')
    plt.subplot(2, 3, 6)
    plt.plot(example['future_states_list'][2, :, 0], label='spacing')
    plt.xlabel('Time step (Ts = 0.04 s)')
    plt.ylabel('Speed (m/s)')

    plt.show()

    
    