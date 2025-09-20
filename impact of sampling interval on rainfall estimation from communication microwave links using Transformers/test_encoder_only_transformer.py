import numpy as np
import pynncml as pnc
import torch
import math
import os
import sys
import torch.nn as nn
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Subset
from io import StringIO
from scipy.signal import savgol_filter

'''
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)'''

def lr_schedule(epoch):
    if epoch < 500:
        return 1.0        # relative to base LR = 1e-4 → stays 1e-4
    elif epoch < 600:
        return 0.5        # 0.5 × 1e-4 = 5e-5
    else:
        return 0.1        # 0.1 × 1e-4 = 1e-5

xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-08-31")  # Time Interval

#samples_type = "min_max"  # Options: "instantaneous", "min_max"
#sampling_interval_in_sec = 900 # Options: 10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900

combinations = [("instantaneous", sec) for sec in [20]]
#combinations.append(("min_max", None))

for samples_type, sampling_interval_in_sec in combinations:

    if samples_type == "min_max":
        dynamic_input_size = 4  # MRSL, mRSL, MTSL, mTSL
        sampling_interval_in_sec = 900
    elif samples_type == "instantaneous":
        dynamic_input_size = 2 * (900 // sampling_interval_in_sec)


    # Set output directory based on sampling configuration (lab computer path)
    base_output_dir = "/home/lucy3/BarakMachlev/Thesis/Final_Results/Transformer/openMRG_DataSet/FineTuning/Set_2"
    if samples_type == "instantaneous":
        output_dir = os.path.join(base_output_dir, f"Instantaneous_{sampling_interval_in_sec}_sec")
    else:
        output_dir = os.path.join(base_output_dir, "Max_Min")

    os.makedirs(output_dir, exist_ok=True)

    dataset = pnc.datasets.loader_open_mrg_dataset(restriction_minimum_length=0.75,
                                                xy_min = xy_min,
                                                xy_max = xy_max,
                                                time_slice = time_slice,
                                                samples_type = samples_type,
                                                sampling_interval_in_sec = sampling_interval_in_sec)

    plt.figure(1)
    dataset.link_set.plot_links(scale=True, scale_factor=1.0)
    plt.grid()
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title("CML Link Map")
    figure_name = "CML_Link_Map.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    plt.tight_layout()
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
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
    figure_name = "Rain_Rate_Histogram.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    batch_size = 16
    lr = 1e-4  # @param{type:"number"}
    weight_decay = 1e-4  # @param{type:"number"}
    n_epochs = 700  # @param{type:"integer"}
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


    # Confirm we have exactly 80 links
    assert len(dataset.link_set.link_list) == 80, f"Expected 80 links, got {len(dataset.link_set.link_list)}"

    # Predefined validation sets
    validation_set_1 = [0, 4, 10, 12, 18, 22, 28, 30, 55, 33, 35, 45, 49, 67, 68, 70]
    validation_set_2 = [5, 9, 16, 34, 39, 40, 42, 47, 54, 31, 56, 61, 62, 64, 73, 79]
    validation_set_3 = [3, 6, 7, 8, 13, 17, 19, 25, 36, 38, 44, 50, 53, 65, 66, 76]
    validation_set_4 = [11, 15, 24, 26, 21, 32, 41, 43, 46, 48, 57, 58, 72, 75, 77, 78]
    validation_set_5 = [1, 2, 14, 20, 27, 23, 29, 37, 51, 52, 59, 60, 63, 69, 71, 74]

    # Choose which validation set to use (1 to 5)
    fold = 2 # 👈 Change this to 2, 3, 4, or 5 for other runs

    # Map fold number to validation set
    val_indices = eval(f"validation_set_{fold}")

    # Confirm 80 links
    assert len(dataset.link_set.link_list) == 80, f"Expected 80 links, got {len(dataset.link_set.link_list)}"

    # Derive training set as the complement
    all_indices = set(range(80))
    train_indices = sorted(list(all_indices - set(val_indices)))

    # Sanity checks
    assert len(val_indices) == 16
    assert len(train_indices) == 64
    assert set(train_indices).isdisjoint(set(val_indices)), "❌ Overlap between training and validation sets!"

    # Create subsets
    training_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    # DataLoaders (deterministic)
    data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Confirm
    print(f"✅ Using only links with length ≥ 750m")
    print(f"✅ Fold {fold} | Train set: {len(train_indices)} links | Val set: {len(val_indices)} links")


    model = pnc.scm.rain_estimation.two_step_network_with_attention(normalization_cfg=pnc.training_helpers.compute_data_normalization(data_loader, network_dynamic_input_size = dynamic_input_size), # Compute the normalization statistics from the training dataset.
                                                                    dynamic_input_size = dynamic_input_size,
                                                                    metadata_input_size = metadata_input_size,
                                                                    d_model = d_model,
                                                                    metadata_n_features = metadata_n_features,
                                                                    window_size = window_size,
                                                                    dropout = dropout,
                                                                    num_encoder_layers = num_encoder_layers,
                                                                    h = h).to(device)

    # Original Hai optimizer - The lr here is a dummy starting point — it will be overridden by the scheduler.
    opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

    class RegressionLoss(torch.nn.Module):
        def __init__(self, in_gamma, gamma_s=0.9):
            super(RegressionLoss, self).__init__()
            self.in_gamma = in_gamma
            self.gamma_s = gamma_s

        def forward(self, input, target):
            delta = (target - input) ** 2
            w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
            return torch.sum(torch.mean(w * delta, dim=0))

    from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis

    # Initialize metrics and results accumulators
    ra = ResultsAccumulator()
    am = AverageMetric()

    # ADD: validation accumulators
    ra_val = ResultsAccumulator()
    am_val = AverageMetric()

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
                    am.add_results(loss=loss.item(),
                                   loss_est=loss_est.item(),
                                   loss_detection=loss_detection.item())  # Log results to average.

            scheduler.step()        
            ra.add_results(loss=am.get_results("loss"),
                           loss_est=am.get_results("loss_est"),
                           loss_detection=am.get_results("loss_detection"))

            ##############################################
            # ----- VALIDATION for this epoch -----
            ##############################################
            model.eval()
            am_val.clear()
            with torch.no_grad():
                for rain_rate_v, rsl_v, tsl_v, metadata_v in val_loader:
                    m_step_v = math.floor(rain_rate_v.shape[1] / window_size)
                    for step_v in range(m_step_v):
                        _rr_v  = rain_rate_v[:, step_v*window_size:(step_v+1)*window_size].float().to(device)
                        _rsl_v = rsl_v[:, step_v*window_size:(step_v+1)*window_size, :].to(device)
                        _tsl_v = tsl_v[:, step_v*window_size:(step_v+1)*window_size, :].to(device)

                        out_v = model(torch.cat([_rsl_v, _tsl_v], dim=-1), metadata_v.to(device))
                        rain_hat_v       = out_v[:, :, 0]
                        rain_detection_v = out_v[:, :, 1]

                        loss_est_v = loss_function_rain_est(rain_hat_v, _rr_v)
                        loss_det_v = loss_function_wet_dry(rain_detection_v, (_rr_v > 0.1).float())
                        loss_v     = lambda_value * loss_est_v + loss_det_v

                        am_val.add_results(loss=loss_v.item(),
                                           loss_est=loss_est_v.item(),
                                           loss_detection=loss_det_v.item())

            # Store validation epoch means
            ra_val.add_results(loss=am_val.get_results("loss"),
                            loss_est=am_val.get_results("loss_est"),
                            loss_detection=am_val.get_results("loss_detection"))
            model.train()  # back to train mode for next epoch
            ##############################################
            # ----- END of VALIDATION for this epoch -----
            ##############################################

        
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

        # 1) Wet/Dry (classification) — train vs val
        plt.figure()
        plt.plot(ra.get_results("loss_detection"),     label="Train Wet/Dry Loss")
        plt.plot(ra_val.get_results("loss_detection"), label="Val Wet/Dry Loss")
        plt.grid()
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Wet/Dry Loss per Epoch ({samples_type}.png)")
        figure_name = f"wet_dry_loss_train_vs_val_{samples_type}.png"
        save_path = os.path.join(output_dir, figure_name)
        plt.savefig(save_path)
        print(f"✅ Figure saved to {save_path}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        # 2) Rain rate (regression) — train vs val
        plt.figure()
        plt.plot(ra.get_results("loss_est"),     label="Train Rain-Rate Loss")
        plt.plot(ra_val.get_results("loss_est"), label="Val Rain-Rate Loss")
        plt.grid()
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Rain-Rate Loss per Epoch ({samples_type}.png)")
        figure_name = f"rain_rate_loss_train_vs_val_{samples_type}.png"
        save_path = os.path.join(output_dir, figure_name)
        plt.savefig(save_path)
        print(f"✅ Figure saved to {save_path}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        print("-----------------------------------------------")
        print(steps_counter)
        print("-----------------------------------------------")

    print("📊 Evaluating model...")
    model.eval()
    #ema_model.eval()
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
    figure_name = f"confusion_matrix_{samples_type}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

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

    start_idx = 115
    end_idx = 205
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