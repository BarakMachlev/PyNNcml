import numpy as np
import pynncml as pnc
import torch
import math
import os
import sys
import torch.nn as nn
import scipy.stats
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import metrics
from io import StringIO


class SyntheticLink:
    def __init__(self, tsl, meta_data):
        self.rsl = None
        self.tsl = tsl
        self.rain_rate = None        # R(t) - exponential
        self.attenuation = None      # A(t) = a * R(t)^b * L
        self.rain_rate_15min = None  # Down sampled to 15-minute averages
        self.meta_data = meta_data

#samples_type = "min_max"  # Options: "instantaneous", "min_max"
#sampling_interval_in_sec = 10   # Options: 10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900

combinations = [("instantaneous", sec) for sec in [10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900]]
combinations.append(("min_max", None))

for samples_type, sampling_interval_in_sec in combinations:

    with open("synthetic_dataset.pkl", "rb") as f:
        synthetic_dataset = pickle.load(f)
        
    if samples_type == "min_max":
        dynamic_input_size = 4  # MRSL, mRSL, MTSL, mTSL
        sampling_interval_in_sec = 900
    elif samples_type == "instantaneous":
        dynamic_input_size = 2 * (900 // sampling_interval_in_sec)

    # Set output directory based on sampling configuration (lab computer path)
    base_output_dir = "/home/lucy3/BarakMachlev/Thesis/Final_Results/Transformer/Synthetic_DataSet/Graph_4/15_sigma"
    if samples_type == "instantaneous":
        output_dir = os.path.join(base_output_dir, f"Instantaneous_{sampling_interval_in_sec}_sec")
    else:
        output_dir = os.path.join(base_output_dir, "Max_Min")

    os.makedirs(output_dir, exist_ok=True)

    for link in synthetic_dataset:
        if samples_type == "instantaneous":
            stride = sampling_interval_in_sec // 10
            link.rsl = link.rsl[::stride]
            link.tsl = link.tsl[::stride]

            feature_len = 900 // sampling_interval_in_sec
            total_samples = len(link.rsl)
            assert total_samples % feature_len == 0, "Length must be divisible by feature count per window"
            link.rsl = link.rsl.reshape(-1, feature_len)  # [T, F]
            link.tsl = link.tsl.reshape(-1, feature_len)  # [T, F]

        elif samples_type == "min_max":
            assert len(link.rsl) % 90 == 0, "Length must be divisible by 90"
            rsl_reshaped = link.rsl.reshape(-1, 90)
            tsl_reshaped = link.tsl.reshape(-1, 90)

            max_rsl = rsl_reshaped.max(axis=1)
            min_rsl = rsl_reshaped.min(axis=1)
            max_tsl = tsl_reshaped.max(axis=1)
            min_tsl = tsl_reshaped.min(axis=1)

            link.rsl = np.stack([max_rsl, min_rsl], axis=1)  # [T, 2]
            link.tsl = np.stack([max_tsl, min_tsl], axis=1)  # [T, 2]

    batch_size = 16
    lr = 1e-4  # @param{type:"number"}
    weight_decay = 1e-4  # @param{type:"number"}
    n_epochs = 30  # @param{type:"integer"}
    window_size = 32
    metadata_n_features = 32
    metadata_input_size = 2
    d_model = 256
    dropout = 0.1
    num_encoder_layers = 4
    h = 8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)
    if device.type == "cuda":
        print("  - GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("  - Running on CPU")

    # Wrapper class
    class SyntheticDatasetWrapper(Dataset):
        def __init__(self, synthetic_links):
            self.synthetic_links = synthetic_links

        def __len__(self):
            return len(self.synthetic_links)

        def __getitem__(self, idx):
            link = self.synthetic_links[idx]
            return (
                torch.tensor(link.rain_rate_15min, dtype=torch.float32),  # [T] - labels
                torch.tensor(link.rsl, dtype=torch.float32),  # [T, F_rsl]
                torch.tensor(link.tsl, dtype=torch.float32),  # [T, F_tsl]
                torch.tensor([link.meta_data.length, link.meta_data.frequency], dtype=torch.float32)
            )

    # Wrap dataset
    wrapped_dataset = SyntheticDatasetWrapper(synthetic_dataset)

    # Fixed split: first 64 for training, last 16 for validation
    train_indices = list(range(64))
    val_indices = list(range(64, 80))

    training_dataset = Subset(wrapped_dataset, train_indices)
    validation_dataset = Subset(wrapped_dataset, val_indices)

    # DataLoaders
    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = pnc.scm.rain_estimation.two_step_network_with_attention(normalization_cfg=pnc.training_helpers.compute_data_normalization(data_loader, network_dynamic_input_size = dynamic_input_size), # Compute the normalization statistics from the training dataset.
                                                                    dynamic_input_size=dynamic_input_size,
                                                                    metadata_input_size=metadata_input_size,
                                                                    d_model=d_model,
                                                                    metadata_n_features=metadata_n_features,
                                                                    window_size=window_size,
                                                                    dropout=dropout,
                                                                    num_encoder_layers=num_encoder_layers,
                                                                    h=h).to(device)

    # Collect all rain rate values across all links
    all_rr = np.concatenate([link.rain_rate for link in synthetic_dataset])
    exp_gamma = scipy.stats.expon.fit(all_rr)[1]  # This is the scale parameter (lambda⁻¹)

    print("Rain Rate Statistics")
    print(f"Mean [mm/hr]: {np.mean(all_rr):.4f}")
    print(f"Std  [mm/hr]: {np.std(all_rr):.4f}")
    print(f"Percentage of wet samples: {100 * np.sum(all_rr > 0) / all_rr.size:.2f}%")
    print(f"Percentage of dry samples: {100 * np.sum(all_rr == 0) / all_rr.size:.2f}%")
    print(f"Fitted exponential scale (lambda⁻¹): {exp_gamma:.4f}")


    opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    class RegressionLoss(torch.nn.Module):
        def __init__(self, in_gamma, gamma_s=0.9):
            super(RegressionLoss, self).__init__()
            self.in_gamma = in_gamma
            self.gamma_s = gamma_s

        def forward(self, input, target):
            delta = (target - input) ** 2
            w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
            return torch.sum(torch.mean(w * delta, dim=0))

    ra = ResultsAccumulator()
    am = AverageMetric()

    model_path = os.path.join(output_dir, "trained_model.pth")

    if os.path.exists(model_path):
        # Load model if weights already exist
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded from: {model_path} — skipping training and loss plotting")

    else:

        print("🟡 No saved weights found — starting training")

        loss_function_rain_est = RegressionLoss(exp_gamma)
        loss_function_wet_dry = torch.nn.BCELoss()

        # Find lambda values such that at initialization both loss will be equal:
        loss_est = 0
        loss_detection = 0
        with torch.no_grad():
            for rain_rate, rsl, tsl, metadata in data_loader:
                m_step = math.floor(rain_rate.shape[1] / window_size)
                for step in range(m_step):
                    _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                    _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                    _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                    rain_estimation_detection = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device))
                    rain_hat = rain_estimation_detection[:, :, 0]
                    rain_detection = rain_estimation_detection[:, :, 1]

                    loss_est += loss_function_rain_est(rain_hat, _rr)
                    loss_detection += loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
        lambda_value = loss_detection / loss_est

        steps_counter=0

        # Train model if weights do not exist
        model.train()
        for epoch in tqdm(range(n_epochs)): # Repeat the whole training process again
            am.clear()
            for rain_rate, rsl, tsl, metadata in data_loader: # for loop for true batches - each batch has batch_size links
                m_step = math.floor(rain_rate.shape[1] / window_size)
                for step in range(m_step): # for loop for sliding windows in time (a.k.a mini-batches / chunks)
                    opt.zero_grad()  # Zero gradients
                    # Perform sliding window in the CML time series.
                    _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                    _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                    _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                    # Forward pass of the model
                    rain_estimation_detection = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device)) # Forward pass
                    rain_hat = rain_estimation_detection[:, :, 0]
                    rain_detection = rain_estimation_detection[:, :, 1]
                    # Compute loss function
                    loss_est = loss_function_rain_est(rain_hat, _rr)
                    loss_detection = loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
                    loss = lambda_value * loss_est + loss_detection
                    # Take the derivative w.r.t. model parameters $\Theta$
                    loss.backward()
                    opt.step()
                    steps_counter += 1
                    am.add_results(loss=loss.item(), loss_est=loss_est.item(),
                                loss_detection=loss_detection.item())  # Log results to average.
            #scheduler.step()
            ra.add_results(loss=am.get_results("loss"), loss_est=am.get_results("loss_est"),
                        loss_detection=am.get_results("loss_detection"))

        # Save trained weights
        torch.save(model.state_dict(), model_path)
        print(f"✅ Weights saved to: {model_path}")

        plt.plot(ra.get_results("loss"), label="Total Loss")
        plt.plot(ra.get_results("loss_est"), label="Rain Rate Loss")
        plt.plot(ra.get_results("loss_detection"), label="Wet/Dry Loss")
        plt.grid()
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss per Epoch ({samples_type}.png)")
        figure_name = f"loss_plot_over_epochs_{samples_type}.png"
        save_path = os.path.join(output_dir, figure_name)
        plt.savefig(save_path)
        print(f"✅ Figure saved to {save_path}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        print("-----------------------------------------------")
        print(steps_counter)
        print("-----------------------------------------------")

    model.eval()
    ga = GroupAnalysis()

    with torch.no_grad():
        for rain_rate, rsl, tsl, metadata in val_loader:
            m_step = math.floor(rain_rate.shape[1] / window_size)
            am.clear()
            rain_ref_list = []
            rain_hat_list = []
            detection_list = []

            for step in range(m_step):
                _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                rain_estimation_detection = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device))
                rain_detection = rain_estimation_detection[:, :, 1]
                rain_hat = rain_estimation_detection[:, :, 0] * torch.round(rain_detection)  # Rain Rate is computed only for wet samples
                rain_hat_list.append(rain_hat.detach().cpu().numpy())
                rain_ref_list.append(_rr.detach().cpu().numpy())
                ga.append(rain_ref_list[-1], rain_hat_list[-1])
                detection_list.append(torch.round(rain_detection).detach().cpu().numpy())
                delta = rain_hat.squeeze(dim=-1) - _rr
                bias = torch.mean(delta)
                mse = torch.mean(delta ** 2) # Dry - Wet MSE
                #mse = torch.mean(delta[_rr > 0] ** 2) # Wet MSE
                #mse = torch.mean(delta[_rr == 0] ** 2)  # Dry MSE
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

    # When there is 100% rain no need to compute confusion matrix

    results_path = os.path.join(output_dir, f"Estimation_Results_{samples_type}.txt")

    # Redirect stdout to capture printed PrettyTable
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("Results Estimation:")
    _ = ga.run_analysis(np.stack([g_array[:-1], g_array[1:]], axis=-1))

    # Restore normal stdout
    sys.stdout = old_stdout

    # Write captured output to file
    with open(results_path, "w") as f:
        f.write(mystdout.getvalue())

    print(f"✅ Results summary saved to: {results_path}")

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
    figure_name = f"Detections_{samples_type}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    rain_hat_array = np.concatenate(rain_hat_list, axis=1)
    rain_ref_array = np.concatenate(rain_ref_list, axis=1)

    plt.plot(np.cumsum(np.maximum(rain_hat_array[-1, :], 0)), label="Two-Steps RNN")
    plt.plot(np.cumsum(rain_ref_array[-1, :]), "--", label="Reference")
    plt.grid()
    plt.legend()
    plt.ylabel("Accumulated Rain-Rate[mm]")
    plt.xlabel("# Samples")
    figure_name = f"Accumulated_Rain_Rate_{samples_type}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    start_idx = 1950
    end_idx = 2150
    x = np.arange(start_idx, end_idx)

    ref = rain_ref_array[-1, start_idx:end_idx]
    hat = rain_hat_array[-1, start_idx:end_idx]

    # Compute Pearson correlation - Avoid zero variance issue
    if np.std(ref) == 0 or np.std(hat) == 0:
        corr = float('nan')  # or corr = 0.0 if you prefer
    else:
        corr = np.corrcoef(ref, hat)[0, 1]

    plt.figure()
    plt.plot(x, ref, label="Reference", linestyle="--")
    plt.plot(x, hat, label="Estimated")
    plt.xlabel("Sample Index")
    plt.ylabel(f"Rain Rate [mm/15 min]")
    plt.title(f"Predicted vs. Reference Rain Rate\nSamples {start_idx}–{end_idx}, Corr = {corr:.2f}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    figure_name = f"RainRate_{samples_type}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()