import numpy as np
import pynncml as pnc
import torch
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from sklearn import metrics

xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-06-10")  # Time Interval

samples_type = "instantaneous"  # "instantaneous", "min_max", "original"
window_size_in_min = 1

if samples_type == "min_max":
    window_size_in_min=15
    rnn_input_size = 4
elif samples_type == "instantaneous":
    rnn_input_size = 2 * (15 // window_size_in_min)
elif samples_type == "original":
    rnn_input_size = 180
else:
    raise ValueError(f"Unknown samples_type: {samples_type}")

dataset = pnc.datasets.loader_open_mrg_dataset(xy_min=xy_min,
                                               xy_max=xy_max,
                                               time_slice=time_slice,
                                               samples_type=samples_type,
                                               window_size_in_min=window_size_in_min)

plt.figure(1)
dataset.link_set.plot_links(scale=True, scale_factor=1.0)
plt.grid()
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.title("CML Link Map")
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(2)
rg = np.stack([p.data_array for p in dataset.point_set]).flatten()
param = scipy.stats.expon.fit(rg)
exp_gamma = param[1]
print("Rain Rate Statistics")
print("Mean[mm/hr]:", np.mean(rg))
print("Std[mm/hr]:", np.std(rg))
print("Parentage of wet samples", 100 * np.sum(rg > 0) / rg.size)
print("Parentage of dry samples", 100 * np.sum(rg == 0) / rg.size)
print("Exponential Distribution Parameters:", param)
_ = plt.hist(rg, bins=100, density=True)
plt.plot(np.linspace(0, np.max(rg), 100), scipy.stats.expon.pdf(np.linspace(0, np.max(rg), 100), *param))
plt.grid()
plt.xlabel("Rain Rate[mm/hr]")
plt.ylabel("Density")
plt.title("Rain Rate Histogram")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

batch_size = 16  # @param{type:"integer"}
window_size = 32  # @param{type:"integer"}
rnn_n_features = 128  # @param{type:"integer"}
metadata_n_features = 32  # @param{type:"integer"}
n_layers = 2  # @param{type:"integer"}
lr = 1e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
rnn_type = pnc.neural_networks.RNNType.GRU  # RNN Type
n_epochs = 100  # @param{type:"integer"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)

model = pnc.scm.rain_estimation.two_step_network(n_layers=n_layers,  # Number of RNN layers
                                                 rnn_type=rnn_type,  # Type of RNN (GRU, LSTM)
                                                 normalization_cfg=pnc.training_helpers.compute_data_normalization(data_loader, rnn_input_size = rnn_input_size), # Compute the normalization statistics from the training dataset.
                                                 rnn_input_size = rnn_input_size,  # 3 + 3 (RSL + TSL)
                                                 rnn_n_features=rnn_n_features,  # Number of features in the RNN
                                                 metadata_input_size=2,  # Number of metadata features
                                                 metadata_n_features=metadata_n_features, # Number of features in the metadata
                                                 pretrained=False).to(device)  # Pretrained model is set to False to train the model from scratch.

class RegressionLoss(torch.nn.Module):
    def __init__(self, in_gamma, gamma_s=0.9):
        super(RegressionLoss, self).__init__()
        self.in_gamma = in_gamma
        self.gamma_s = gamma_s

    def forward(self, input, target):
        delta = (target - input) ** 2
        w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
        return torch.sum(torch.mean(w * delta, dim=0))


loss_function_rain_est = RegressionLoss(exp_gamma)
loss_function_wet_dry = torch.nn.BCELoss()

# Find lambda values such that at initialization both loss will be equal:
loss_est = 0
loss_detection = 0
with torch.no_grad():
    for rain_rate, rsl, tsl, metadata in data_loader:
        state = model.init_state(batch_size=rsl.shape[0])
        m_step = math.floor(rain_rate.shape[1] / window_size)
        for step in range(m_step):
            _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
            _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            rain_estimation_detection, state = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device),
                                                     state.detach())
            rain_hat = rain_estimation_detection[:, :, 0]
            rain_detection = rain_estimation_detection[:, :, 1]

            loss_est += loss_function_rain_est(rain_hat, _rr)
            loss_detection += loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
lambda_value = loss_detection / loss_est


from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis

opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
ra = ResultsAccumulator()
am = AverageMetric()

model.train()
print("Start Training")
for epoch in tqdm(range(n_epochs)): # Repeat the whole training process again
    am.clear()
    for rain_rate, rsl, tsl, metadata in data_loader: # for loop for true batches - each batch has batch_size links
        state = model.init_state(batch_size=rsl.shape[0])
        m_step = math.floor(rain_rate.shape[1] / window_size)
        for step in range(m_step): # for loop for sliding windows in time (a.k.a mini-batches / chunks)
            opt.zero_grad()  # Zero gradients
            # Perform sliding window in the CML time series.
            _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
            _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            # Forward pass of the model
            rain_estimation_detection, state = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device), state.detach()) # Forward pass
            rain_hat = rain_estimation_detection[:, :, 0]
            rain_detection = rain_estimation_detection[:, :, 1]
            # Compute loss function
            loss_est = loss_function_rain_est(rain_hat, _rr)
            loss_detection = loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
            loss = lambda_value * loss_est + loss_detection
            # Take the derivative w.r.t. model parameters $\Theta$
            loss.backward()
            opt.step()
            am.add_results(loss=loss.item(), loss_est=loss_est.item(),
                           loss_detection=loss_detection.item())  # Log results to average.
    ra.add_results(loss=am.get_results("loss"), loss_est=am.get_results("loss_est"),
                   loss_detection=am.get_results("loss_detection"))

plt.plot(ra.get_results("loss"), label="Total Loss")
plt.plot(ra.get_results("loss_est"), label="Rain Rate Loss")
plt.plot(ra.get_results("loss_detection"), label="Wet/Dry Loss")
plt.grid()
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss per Epoch ({samples_type} sampling)")
plt.savefig(f"/Users/barakmachlev/Documents/Thesis/Results/{samples_type}/loss_plot_over_epochs_{samples_type}.png")
plt.show(block=False)
plt.pause(5)
plt.close()


model.eval()
ga = GroupAnalysis()
with torch.no_grad():
    for rain_rate, rsl, tsl, metadata in val_loader:
        state = model.init_state(batch_size=rsl.shape[0])
        m_step = math.floor(rain_rate.shape[1] / window_size)
        am.clear()
        rain_ref_list = []
        rain_hat_list = []
        detection_list = []

        for step in range(m_step):
            _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
            _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            rain_estimation_detection, state = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device), state.detach())
            rain_detection = rain_estimation_detection[:, :, 1]
            rain_hat = rain_estimation_detection[:, :, 0] * torch.round(rain_detection)  # Rain Rate is computed only for wet samples
            rain_hat_list.append(rain_hat.detach().cpu().numpy())
            rain_ref_list.append(_rr.detach().cpu().numpy())
            ga.append(rain_ref_list[-1], rain_hat_list[-1])
            detection_list.append(torch.round(rain_detection).detach().cpu().numpy())
            delta = rain_hat.squeeze(dim=-1) - _rr
            bias = torch.mean(delta)
            mse = torch.mean(delta ** 2)
            am.add_results(bias=bias.item(), mse=mse.item())
actual = np.concatenate(detection_list).flatten()
predicted = (np.concatenate(rain_ref_list) > 0.1).astype("float").flatten()
confusion_matrix = metrics.confusion_matrix(actual, predicted)
max_rain = np.max(np.concatenate(rain_ref_list))
g_array = np.linspace(0, max_rain, 6)

print("Results Detection:")
print("Validation Results of Two-Step RNN")
print("Accuracy[%]:", 100 * (np.sum(actual == predicted) / actual.size))
print("F1 Score:", metrics.f1_score(actual, predicted))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

cm_display.plot()
plt.title(f"Confusion Matrix ({samples_type} Sampling)")
plt.savefig(f"/Users/barakmachlev/Documents/Thesis/Results/{samples_type}/confusion_matrix_{samples_type}.png")
plt.show(block=False)
plt.pause(5)
plt.close()

print("Results Estimation:")
_ = ga.run_analysis(np.stack([g_array[:-1], g_array[1:]], axis=-1))

detection_array = np.concatenate(detection_list, axis=1)
rain_ref_array = np.concatenate(rain_ref_list, axis=1)
detection_array = np.round(detection_array)

rain_array = rain_ref_array[0, :300]
detection_array = detection_array[0, :300]
fig, ax = plt.subplots()
x = np.arange(rain_array.shape[0])
ax.plot(x, rain_array, label="Rain")
rain_max = np.max(rain_array)
ax.fill_between(x, rain_max, where=np.logical_or(np.logical_and(detection_array == 1, rain_array > 0),
                                                 np.logical_and(detection_array == 0, rain_array == 0)),
                facecolor='green', alpha=.5, label="Detection")
ax.fill_between(x, rain_max, where=np.logical_and(detection_array == 0, rain_array > 0), facecolor='red',
                alpha=.5, label="Mis-Detection")
ax.fill_between(x, rain_max, where=np.logical_and(detection_array == 1, rain_array == 0), facecolor='blue',
                alpha=.5, label="False Alarm")
plt.legend()
plt.ylabel("Rain Rate [mm/hr]")
plt.xlabel("Sample Index")
plt.grid()
plt.show(block=False)
plt.pause(5)
plt.close()

rain_hat_array = np.concatenate(rain_hat_list, axis=1)
rain_ref_array = np.concatenate(rain_ref_list, axis=1)

plt.plot(np.cumsum(np.maximum(rain_hat_array[0, :], 0)), label="Two-Steps RNN")
plt.plot(np.cumsum(rain_ref_array[0, :]), "--", label="Reference")
plt.grid()
plt.legend()
plt.ylabel("Accumulated Rain-Rate[mm]")
plt.xlabel("# Samples")
plt.savefig(f"/Users/barakmachlev/Documents/Thesis/Results/{samples_type}/Accumulated_Rain_Rate_{samples_type}.png")
plt.show()